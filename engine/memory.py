"""Per-process driver memory accounting, safe for a shared GPU."""
import os
import subprocess
import threading
from pathlib import Path

import torch

_driver_identity = None


def driver_processes():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=10)


def initialize_process_memory():
    """Resolve driver PID across namespaces around creation of our CUDA context.

    Accept a direct PID match or exactly one new process on the selected GPU.
    If another job starts simultaneously and makes this ambiguous, fail closed.
    """
    global _driver_identity
    uuid = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).strip()
    def pids(rows):
        return {int(parts[1]) for line in rows.splitlines()
                if (parts := [p.strip() for p in line.split(',')])[0] == uuid}
    before = pids(driver_processes())
    torch.cuda.init()
    allocation = torch.zeros(1, device='cuda:0')
    torch.cuda.synchronize()
    after = pids(driver_processes())
    own = {os.getpid()}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('NSpid:'):
            own.update(map(int, line.split()[1:]))
    candidates = after & own
    method = 'PID/NSpid match'
    if not candidates:
        candidates = after - before
        method = 'Unique new driver PID across own CUDA context creation'
    if len(candidates) != 1:
        raise RuntimeError(f'Ambiguous driver PID mapping: before={before}, after={after}, own={own}')
    _driver_identity = (uuid, candidates, method)
    del allocation


def process_mib(output, gpu_uuid, pids):
    values = []
    for line in output.splitlines():
        uuid, pid, memory = [part.strip() for part in line.split(',')]
        if uuid == gpu_uuid and int(pid) in pids:
            values.append(float(memory))
    return sum(values) if values else None


class MemoryWatch:
    def __init__(self):
        if _driver_identity is None:
            raise RuntimeError('initialize_process_memory must run before GPU initialization')
        self.uuid, self.pids, self.identity_method = _driver_identity
        self.peak_mib = 0.; self.samples = 0; self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def sample(self):
        output = driver_processes()
        value = process_mib(output, self.uuid, self.pids)
        if value is None:
            raise RuntimeError(f'Cannot identify own GPU process: {sorted(self.pids)}')
        self.peak_mib = max(self.peak_mib, value); self.samples += 1

    def watch(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
            except Exception as error:
                self.errors.append(repr(error))
            self.stop_event.wait(1)

    def __enter__(self):
        self.sample()  # Fail closed if driver PID cannot be identified.
        self.thread.start()
        return self

    def report(self):
        return dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            sampled_process_peak_gib=self.peak_mib/1024, process_samples=self.samples,
            driver_pids=sorted(self.pids), pid_resolution=self.identity_method,
            measurement='Driver memory for this process only; other GPU users excluded.',
            sampling_interval_seconds=1, sampling_errors=self.errors)

    def __exit__(self, *args):
        self.stop_event.set(); self.thread.join(timeout=12)


def check_memory(report):
    if max(report['peak_reserved_gib'], report['sampled_process_peak_gib']) >= 16:
        raise RuntimeError('16 GiB process memory budget exceeded; do not reduce batch automatically')
    if not report['process_samples']:
        raise RuntimeError('No driver-level process memory observation')
