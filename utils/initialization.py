"""Original untrained encoders and deterministic isolated Transformer/LoRA init."""
import json
import torch
from engine.audit import digest, file_digest
from utils.protocol import ROOT, REFERENCE, SOURCE


def load_task_table(task):
    from utils.task_projection import load_task_table as load_raw_table
    return load_raw_table(task)


def load_position_encoding():
    path = ROOT / 'artifacts/depth2_video_adaptive_fixed_position.pt'
    audit = json.loads(path.with_suffix('.json').read_text())
    if file_digest(path) != audit['snapshot_sha256'] or audit['trained_parameters_imported'] != 0:
        raise ValueError('Fixed position snapshot changed')
    state = torch.load(path,map_location='cpu',weights_only=True)
    if set(state) != {'fusion.position_encoding'} or digest(state) != audit['tensor_sha256']:
        raise ValueError('Fixed position tensor changed')
    return state['fusion.position_encoding']


def restore_encoder_initialization(model,config):
    snapshot = torch.load(config['training']['initialization'],map_location='cpu',weights_only=True,mmap=True)
    if snapshot['seed'] != 42:
        raise ValueError('Expected original untrained seed42 initialization')
    prefixes = ('technical_expert.','motion_position_embedding.','blip_temporal.')
    saved = {k:v for k,v in snapshot['state'].items() if k.startswith(prefixes)}
    current = {k:v for k,v in model.state_dict().items() if k.startswith(prefixes)}
    if saved.keys() != current.keys() or any(current[k].shape != v.shape or current[k].dtype != v.dtype for k,v in saved.items()):
        raise ValueError('Untrained encoder coverage/shape mismatch')
    model.load_state_dict(saved,strict=False)
    torch.set_rng_state(snapshot['cpu_rng_state'])
    return len(saved)


def initialize_model(config,build_backbones=True):
    from models.aider import AIDER
    from models.lora import task_adapter_items
    from utils.config import seed_everything
    seed_everything(42)
    model = AIDER(config,build_backbones=build_backbones)
    model.train(True)
    reference = json.loads((REFERENCE / 'retained_initialization.json').read_text())
    retained = {k:v for k,v in model.compact_state_dict().items()
                if not k.startswith(tuple(reference['excluded_prefixes']))}
    retained_hash = digest(retained)
    rng = digest({'cpu_rng':torch.get_rng_state()})
    if retained_hash != reference['retained_state_sha256'] or len(retained) != reference['retained_tensors']:
        raise AssertionError('Retained non-fusion modules differ from untrained parent')
    if rng != reference['construction_cpu_rng_sha256']:
        raise AssertionError('Construction RNG differs from parent')
    adapters = list(task_adapter_items(model))
    if len(adapters) != 48 or sum(p.numel() for _,_,a in adapters for p in a.parameters()) != 1179648:
        raise AssertionError('Expected four independent 294,912-parameter LoRA sets')
    pointers = [p.data_ptr() for _,_,a in adapters for p in a.parameters()]
    if len(pointers) != len(set(pointers)) or any(torch.count_nonzero(a.B.weight) for _,_,a in adapters):
        raise AssertionError('LoRA aliasing or nonzero initial delta')
    model.construction_rng_sha256 = rng
    model.initialization_audit.update(initial_state_sha256=digest(model.compact_state_dict()),
        construction_rng_sha256=rng,source=str(SOURCE),trained_checkpoint_loaded=False,
        retained_state_sha256=retained_hash,retained_tensors=len(retained),retained_modules_equal_parent=True,
        shared_transformer_layers=2,gate_count=0,task_lora_parameters=1179648,
        lora_parameters_per_task=294912,task_embedding_transform=config['model']['task_embedding_transform'])
    return model
