"""Untrained seed42 initialization; shared FiLM is audited against old private copies."""
import json
import torch
from engine.audit import digest,file_digest
from utils.protocol import ROOT,REFERENCE,ARTIFACTS,SOURCE


def load_task_table(task):
    from utils.task_projection import load_task_table as load_raw_table
    return load_raw_table(task)


def load_position_encoding():
    path = ARTIFACTS / 'depth2_video_adaptive_fixed_position.pt'
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


def expand_shared_film_for_audit(model):
    """Audit other initial weights using the historical raw task buffer only here."""
    state = model.compact_state_dict()
    # The only intentional model-state change is the final task embedding.
    # Restore its original raw value in this audit view, never in the model.
    state['dimension_encoder.encoded_feature'] = model.dimension_encoder.original_feature
    owners = {name:module for name,module in model.named_modules() if name in (
        'technical_projector.affine','motion_projector.affine','tv_projector.affine','alpha_gate')}
    expanded,inserted = {},set()
    for name,value in state.items():
        owner = next((key for key in owners if name.startswith(key+'.')),None)
        if owner is None:
            expanded[name] = value
        elif owner not in inserted:
            for task in model.task_order:
                for suffix,tensor in owners[owner].state_dict().items():
                    expanded[owner+'.by_task.'+task+'.'+suffix] = tensor
            inserted.add(owner)
    if len(inserted) != 4:
        raise AssertionError('Expected three shared task-FiLM maps and one shared alpha map')
    return expanded


def initialize_model(config,build_backbones=True):
    from models.aider import AIDER
    from models.lora import adapter_items
    from utils.config import seed_everything
    seed_everything(42)
    model = AIDER(config,build_backbones=build_backbones)
    model.train(True)
    reference = json.loads((REFERENCE / 'retained_initialization.json').read_text())
    expanded = {k:v for k,v in expand_shared_film_for_audit(model).items()
                if not k.startswith(tuple(reference['excluded_prefixes']))}
    retained_hash = digest(expanded)
    rng = digest({'cpu_rng':torch.get_rng_state()})
    if retained_hash != reference['retained_state_sha256'] or len(expanded) != reference['retained_tensors']:
        raise AssertionError('Retained modules/shared FiLM initial values differ from parent')
    baseline = json.loads((ROOT / 'baseline_initialization.json').read_text())
    base_hash = digest(dict(model.fusion.base_parameters()))
    if rng != reference['construction_cpu_rng_sha256'] or base_hash != baseline['fusion_base_sha256']:
        raise AssertionError('Shared Transformer base or construction RNG differs from parent')
    adapters = list(adapter_items(model))
    count = sum(p.numel() for _,a in adapters for p in a.parameters())
    if len(adapters) != 12 or count != 368736:
        raise AssertionError('Expected twelve shared 30,728-parameter conditional LoRAs')
    pointers = [p.data_ptr() for _,a in adapters for p in a.parameters()]
    if len(pointers) != len(set(pointers)):
        raise AssertionError('Different projection adapters unexpectedly alias weights')
    for _,a in adapters:
        if any(torch.count_nonzero(p) for p in (a.B.weight,a.gate.weight,a.gate.bias)):
            raise AssertionError('LoRA B and the channel-residual Linear must start at zero')
        condition = model.dimension_encoder.encoded_feature
        if (torch.count_nonzero(a.channel_residual(condition))
                or not torch.equal(a.channel_weights(condition),torch.ones(len(condition),a.rank))):
            raise AssertionError('Initial channel residual must be zero and effective channel weight one')
    if any('.by_task.' in name or '.adapters.' in name for name,_ in model.named_parameters()):
        raise AssertionError('Task-private FiLM/LoRA parameter bank still present')
    model.construction_rng_sha256 = rng
    model.initialization_audit.update(initial_state_sha256=digest(model.compact_state_dict()),
        construction_rng_sha256=rng,source=str(SOURCE),trained_checkpoint_loaded=False,
        retained_state_sha256=retained_hash,retained_tensors=len(expanded),retained_modules_equal_parent=True,
        shared_transformer_base_sha256=base_hash,shared_transformer_base_equal_parent=True,
        shared_transformer_layers=2,router_count=0,channel_modulators=12,
        shared_lora_parameters=294912,channel_modulator_parameters=73824,
        shared_lora_and_gate_parameters=count,shared_task_film_maps=3,shared_alpha_generators=1,
        private_heads=4,task_embedding_transform=config['model']['task_embedding_transform'])
    model.initialization_audit.update(intent_adapter_used=False,
        task_embeddings_are_unmodified_raw_clip=True,
        task_embedding_row_norms=model.dimension_encoder.encoded_feature.norm(dim=1).tolist(),
        lora_modulation='one_plus_linear768to8',lora_gate_weight_init='zeros',lora_gate_bias_init='zeros',
        lora_channel_residual_initial_value=0.0,lora_channel_weight_initial_value=1.0)
    return model
