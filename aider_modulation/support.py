"""Model-only constants/configuration; no dataset, optimizer or experiment runtime."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CONFIGS=ROOT/'configs'


def digest(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(f'{name}|{value.dtype}|{tuple(value.shape)}'.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _read_constant(name):
    path=CONFIGS/name
    expected=json.loads((CONFIGS/'checksums.json').read_text())[name]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Model constant changed: '+name)
    return json.loads(path.read_text())


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load_config(pretrained_root=None,initialization_path=None):
    """Assemble the baseline model; no trained weights are loaded by this function.

    pretrained_root supplies the three external backbone weights and local tokenizer.
    initialization_path optionally restores the original untrained seed42 feature
    projectors/BLIP-SA initialization, for exact historical initialization matching.
    Neither pretrained weights nor initialization snapshots are distributed here.
    """
    arch=_read_constant('model.json');table=_read_constant('task_embeddings.json')
    pretrained=Path(pretrained_root).resolve() if pretrained_root is not None else ROOT.parent/'pretrained_models'
    model=arch['model']
    for block,key in [('vital','model_dir'),('slowfast','checkpoint_path'),('blip','checkpoint_path'),('blip','bert_path')]:
        model['backbones'][block][key]=str(pretrained/model['backbones'][block][key])
    model['backbones']['blip']['med_config']=str(CONFIGS/'med_config.json')
    model['dimension_encoder']['model_dir']=str(pretrained/model['dimension_encoder']['model_dir'])
    # These small metadata keys retain the original constructor/state-dict ABI.
    # They are not optimizer, dataset, schedule or training-launch configuration.
    return dict(model=model,task=dict(order=table['task_order'],descriptions=table['descriptions']),
        training=dict(seed=arch['seed'],initialization=str(Path(initialization_path).resolve()) if initialization_path else None),
        training_strategy='two_stage',experiment=dict(variant='full',contract='aider_modulation_core_v1'),
        ablation=dict(depth=2,tasks=table['task_order']))


def load_task_table(task):
    table=_read_constant('task_embeddings.json')
    if table['task_order']!=task['order'] or table['descriptions']!=task['descriptions']:
        raise ValueError('Task registry/descriptions differ from the fixed raw CLIP table')
    if table['mean_subtraction'] or table['row_l2_normalization']:
        raise ValueError('Expected raw CLIP projections')
    values={name:torch.tensor(table[key],dtype=torch.float32) for name,key in
        [('encoded_feature','transformed'),('original_feature','original'),('task_mean','mean'),('centered_norms','centered_norms')]}
    for name,value in values.items():
        if digest({name:value})!=table['hashes'][name]:raise ValueError('Invalid task constant: '+name)
    if not torch.equal(values['encoded_feature'],values['original_feature']):raise ValueError('Task projections were transformed')
    return values


def load_position_encoding():
    saved=_read_constant('position_encoding.json');position=torch.tensor(saved['values'],dtype=torch.float32)
    if list(position.shape)!=saved['shape'] or tuple(position.shape)!=(1,8,1536):raise ValueError('Invalid position encoding')
    return position


def restore_encoder_initialization(model,config):
    path=config['training'].get('initialization')
    if path is None:return 0
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if saved['seed']!=42:raise ValueError('Expected external untrained seed42 initialization snapshot')
    prefixes=('technical_expert.','motion_position_embedding.','blip_temporal.')
    tensors={k:v for k,v in saved['state'].items() if k.startswith(prefixes)}
    current={k:v for k,v in model.state_dict().items() if k.startswith(prefixes)}
    if current.keys()!=tensors.keys() or any(current[k].shape!=v.shape or current[k].dtype!=v.dtype for k,v in tensors.items()):
        raise ValueError('Untrained initialization coverage/shape mismatch')
    model.load_state_dict(tensors,strict=False);torch.set_rng_state(saved['cpu_rng_state'])
    return len(tensors)


def film_video_condition(config):return config['model']['film_video_condition']


def validate_model_config(config):
    from .registry import TASKS
    m=config['model'];spec=m['readout_spec'];q=m['qk_diagonal']
    if tuple(config['task']['order'])!=TASKS or m['head_policy']!='private':raise ValueError('Expected four private scoring heads')
    if m['film_video_condition']!='video_only' or m['task_film_policy']!='shared':raise ValueError('Expected shared video-conditioned FiLM')
    if (spec['width'],spec['depth'],spec['heads'],spec['ffn_hidden'])!=(1536,2,12,1536):raise ValueError('Baseline Fusion shape changed')
    if q['channels']!=128 or not q['unit_offset'] or q['gate']!='Linear(768,128)':raise ValueError('Expected residual QK diagonal gate')
    if config['training_strategy']!='two_stage' or config['training']['seed']!=42:raise ValueError('Expected baseline construction metadata')
