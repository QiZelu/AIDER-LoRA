"""Independent AIGVQA ordinary Transformer + private task LoRA experiments."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode = True


def startup_memory(gpu):
    values = subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    used,free = map(int,values.split(','))
    if used or free < 16384:
        raise RuntimeError('Selected GPU must be idle with 16 GiB free; no eviction, reassignment, batch reduction or automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','preflight','train','evaluate','summarize'),default='train')
    parser.add_argument('--variant',choices=('joint','two_stage'),default='joint')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--background',action='store_true')
    args = parser.parse_args()
    gpu = args.gpu if args.gpu is not None else {'joint':0,'two_stage':3}[args.variant]
    if not 0 <= gpu <= 7:
        parser.error('Expected physical GPU 0..7')
    if bool(args.checkpoint) != (args.mode == 'evaluate'):
        parser.error('--checkpoint is required only for evaluate')
    if args.background and args.mode != 'train':
        parser.error('--background applies only to train')
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.mode in ('prepare','summarize') else str(gpu)
    os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from utils.protocol import OUTPUTS,prepare,load_prepared,selected_source_hashes
    from engine.io import write_json
    root = (args.output_dir or OUTPUTS[args.variant]).resolve()
    if args.mode in ('preflight','train','evaluate'):
        try:
            startup_memory(gpu)
        except Exception as error:
            write_json(root.with_name(root.name+'.startup_failure.json'),dict(error=repr(error),gpu=gpu,automatic_retry=False))
            raise
    if args.background:
        log = root.with_name(root.name+'.train.worker.log')
        launch = root.with_name(root.name+'.train.launch.json')
        if log.exists() or launch.exists() or any((root / name).exists() for name in ('.training_claim','history.json','best.pt','last.pt')):
            raise FileExistsError('Existing launch/training; refuse overwrite or automatic retry')
        prepare(root,gpu,args.variant)
        command = [sys.executable,'-u','-B',str(Path(__file__).resolve()),'--mode','train',
                   '--variant',args.variant,'--gpu',str(gpu),'--output-dir',str(root)]
        with log.open('x') as handle:
            process = subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        info = dict(pid=process.pid,gpu=gpu,variant=args.variant,command=command,log=str(log),output_dir=str(root),
            process_memory_limit_gib=16,automatic_retry=False,total_epochs=10,total_updates=2310,
            lora=dict(rank=8,alpha=16,dropout=.05),loss='fp32_smooth_l1_beta0.1_no_mi',
            pipeline='disposable phase-aware real batch8 preflight -> fresh seed42 -> 10 epochs -> best/last reload evaluation')
        write_json(launch,info)
        print(info,flush=True)
        return
    from engine.trainer import train,summarize
    if args.mode == 'summarize':
        print(summarize(root))
        return
    if args.mode == 'evaluate':
        import torch
        from utils.initialization import initialize_model
        from engine.memory import initialize_process_memory,MemoryWatch,check_memory
        from engine.checkpoint import load_model_checkpoint
        from engine.evaluation import evaluate,export
        from engine.diagnostics import lora_diagnostics,task_coefficients
        from dataset.loader import build_loader
        config = load_prepared(root)
        if config['training_strategy'] != args.variant or config['experiment']['physical_gpu'] != gpu:
            raise ValueError('Evaluate variant/GPU differs from prepared identity')
        folder = root / 'evaluations' / args.checkpoint.stem
        folder.mkdir(parents=True,exist_ok=False)
        torch.set_num_threads(2)
        initialize_process_memory()
        torch.cuda.set_per_process_memory_fraction(14*1024**3/torch.cuda.get_device_properties(0).total_memory)
        with MemoryWatch() as watcher:
            sources = selected_source_hashes(config)
            model = initialize_model(config)
            model.source_hashes = sources
            payload = load_model_checkpoint(model,args.checkpoint,config)
            epoch = payload['epoch']
            del payload
            model.cuda()
            rows,metrics,diagnostics = evaluate(model,build_loader(config,'val'),torch.device('cuda:0'))
            export(folder,'evaluation',rows,metrics,diagnostics,epoch,args.checkpoint)
            write_json(folder / 'lora.json',lora_diagnostics(model))
            write_json(folder / 'task_coefficients.json',task_coefficients(model))
            check_memory(watcher.report())
            write_json(folder / 'process_memory.json',watcher.report())
        return
    config = prepare(root,gpu,args.variant)
    if args.mode == 'prepare':
        print('Data, raw task table and read-only Priors verified; no training:',root)
        return
    train(config,args.mode)


if __name__ == '__main__':
    main()
