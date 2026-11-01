"""Verify fresh two-stage/joint raw CLIP initialization in isolated CPU processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CODE=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def snapshot(kind,config_path,output):
    import yaml
    config=yaml.safe_load(config_path.read_text())
    resource=Path(config['experiment']['resource_root'])
    os.environ.update(AIDER_RESOURCE_ROOT=str(resource),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    if kind=='joint':
        provenance=read(CODE/'SOURCE_PROVENANCE.json')
        control=Path(config['protocol']['comparison_control'])
        config=yaml.safe_load((control/'config.yaml').read_text())
        for name,item in read(control/'sources.json').items():
            if name.startswith('runtime/') and hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Joint runtime changed: '+name)
        sys.path.insert(0,provenance['source_runtime'])
    else:
        sys.path.insert(0,str(CODE))
    import torch
    from utils.initialization import initialize_model
    from engine.audit import digest
    torch.set_num_threads(2)
    model=initialize_model(config,build_backbones=False)
    state=model.compact_state_dict()
    rng=digest({'cpu_rng':torch.get_rng_state()})
    model.eval()
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(1300042)
        raw={name:torch.randn(4,8,width) for name,width in [('technical',512),('motion',256),('text_video',768)]}
        prediction=model.forward_from_raw(raw,torch.arange(4))['predictions'].tolist()
    write(output,dict(kind=kind,parameter_hash=digest(dict(model.named_parameters())),
        parameter_tensors=len(list(model.parameters())),state_sha256=digest(state),
        construction_rng_sha256=model.construction_rng_sha256,current_cpu_rng_sha256=rng,
        state_tensor_sha256={k:digest({k:v}) for k,v in state.items()},initial_predictions=prediction,
        task_embedding_norms=model.dimension_encoder.encoded_feature.norm(dim=1).tolist(),
        initial_phase=model.fusion.phase,
        trained_checkpoint_loaded=model.initialization_audit['trained_checkpoint_loaded'],build_backbones=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--snapshot',choices=('two_stage','joint'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.snapshot:
        if args.output is None:parser.error('--output is required for snapshots')
        snapshot(args.snapshot,args.config,args.output);return
    if args.output_dir is None:parser.error('--output-dir is required')
    folder=args.output_dir;folder.mkdir(parents=True,exist_ok=False)
    for kind in ('two_stage','joint'):
        with (folder/f'{kind}.log').open('x') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--config',str(args.config),
                '--snapshot',kind,'--output',str(folder/f'{kind}_initialization.json')],
                check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    staged,control=[read(folder/f'{kind}_initialization.json') for kind in ('two_stage','joint')]
    for key in ('parameter_hash','parameter_tensors','state_sha256','state_tensor_sha256','construction_rng_sha256','current_cpu_rng_sha256','task_embedding_norms'):
        if staged[key]!=control[key]:raise AssertionError('Fresh two-stage/joint initialization differs: '+key)
    if staged['initial_phase']!='dense' or control['initial_phase']!='joint':raise AssertionError('Wrong initial phases')
    error=max(abs(a-b) for a,b in zip(staged['initial_predictions'],control['initial_predictions']))
    if error>1e-6:raise AssertionError('Fresh two-stage/joint initial predictions differ')
    if staged['trained_checkpoint_loaded'] or control['trained_checkpoint_loaded']:raise AssertionError('Trained checkpoint used')
    write(folder/'paired_initialization_audit.json',dict(passed=True,non_backbone_parameters_identical=True,
        all_state_tensors_identical=True,all_other_state_tensors_identical=True,model_rng_identical=True,
        initial_prediction_max_abs_error=error,parameter_tensors=staged['parameter_tensors'],
        task_embedding_norms=staged['task_embedding_norms'],initial_phases=['dense','joint'],
        trained_checkpoint_loaded=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('Paired two-stage/joint fresh initialization verified; prediction error',error,flush=True)


if __name__=='__main__':main()
