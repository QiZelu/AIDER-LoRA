#!/usr/bin/env python3
"""Standalone Full-baseline 5+5 training; nothing launches without --mode train."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check','preflight','train','evaluate','summarize'), default='check')
    parser.add_argument('--config', default='configs/train.yaml', help='Relative paths resolve against this package root')
    parser.add_argument('--output-dir', help='Fresh output directory (train/preflight/check)')
    parser.add_argument('--run-dir', help='Existing run directory (evaluate/summarize)')
    parser.add_argument('--gpu', type=int, default=0, help='Physical CUDA index; only this GPU is made visible')
    parser.add_argument('--checkpoint', help='Checkpoint from this standalone trainer (evaluate only)')
    args = parser.parse_args()
    if args.gpu < 0: parser.error('--gpu must be nonnegative')
    if args.checkpoint and args.mode != 'evaluate': parser.error('--checkpoint is evaluation-only')
    if args.mode in ('evaluate','summarize'):
        if not args.run_dir or args.output_dir: parser.error('evaluate/summarize require --run-dir, without --output-dir')
    elif args.run_dir: parser.error('--run-dir is only for evaluate/summarize')
    if args.mode == 'evaluate' and not args.checkpoint: parser.error('evaluate requires --checkpoint')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.mode in ('train','preflight','evaluate') else ''
    for key,value in dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false').items():
        os.environ.setdefault(key,value)
    from training.protocol import make_config, validate_data, audit_priors, prepare_run, load_prepared, resolve
    if args.mode in ('train','preflight','evaluate'):
        free = int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
        if free < 16384: raise RuntimeError('Selected GPU needs at least 16 GiB free; no other processes are stopped')
    if args.mode in ('evaluate','summarize'):
        cfg = load_prepared(resolve(args.run_dir))
    else:
        if args.mode in ('train','preflight'):
            import yaml
            options = yaml.safe_load(resolve(args.config).read_text())
            output = resolve(args.output_dir or options['output_dir'])
            if output.exists(): raise FileExistsError(f'Refuse to overwrite existing run: {output}')
        print('Validating configuration and hashing external resources...',flush=True)
        cfg = make_config(args.config,args.output_dir,args.gpu)
    if args.mode == 'check':
        data = validate_data(cfg)
        print(f"Dataset verified: {data['train_videos']} train / {data['test_videos']} test videos, four tasks.",flush=True)
        print('Checking read-only Prior content hashes...',flush=True)
        prior = audit_priors(cfg)
        print(f"Passed: {prior['videos']} Prior caches; {cfg['counts']['steps_per_epoch']} updates/epoch; 10 epochs; 5+5 phases; warmup {cfg['training']['warmup_steps']} updates.")
        print('No training started and no output directory created.')
    elif args.mode in ('train','preflight'):
        prepare_run(cfg)
        from training.engine.trainer import train
        train(cfg,args.mode)
    elif args.mode == 'evaluate':
        from training.engine.evaluate_one import evaluate_one
        evaluate_one(cfg,resolve(args.checkpoint),args.gpu)
    else:
        from training.engine.trainer import summarize_run
        report = summarize_run(resolve(args.run_dir))
        print(f"Complete: 10 epochs, {report['optimizer_updates']} updates; best epoch {report['best_epoch']}.")


if __name__ == '__main__':
    main()
