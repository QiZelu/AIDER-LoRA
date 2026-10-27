"""Strict complete fusion/adapter checkpoints, including frozen stage-two base."""
import os
from pathlib import Path
import torch
from engine.losses import quality_loss_protocol
from utils.protocol import CONTRACT, validate_config, phase_for_epoch

FROZEN = ('motion_backbone.feature_extraction.','vital_backbone.vision_model.','tv_backbone.model.')


def validate_phase(config,phase,epoch):
    allowed = ('joint',) if config['training_strategy'] == 'joint' else ('dense','adapter')
    if phase not in allowed or (epoch >= 0 and phase != phase_for_epoch(config,epoch)):
        raise ValueError('Checkpoint phase/epoch is inconsistent with the training strategy')


def save_checkpoint(path,model,config,epoch,optimizer=None,scheduler=None,**metadata):
    validate_config(config)
    validate_phase(config,model.fusion.phase,epoch)
    state = model.compact_state_dict()
    if model.tv_backbone is not None:
        prefix = 'tv_backbone.model.text_encoder.'
        if not all(prefix+name in state for name in model.tv_backbone.model.text_encoder.state_dict()):
            raise ValueError('Complete fine-tuned BLIP text is required')
    if any('fusion.'+name not in state for name,_ in model.fusion.named_parameters()):
        raise ValueError('All fusion base and task-adapter weights must be saved, including frozen weights')
    payload = dict(model_contract=CONTRACT,protocol=config['protocol'],config=config,
        loss_protocol=quality_loss_protocol(config),model=state,fusion_phase=model.fusion.phase,
        task_registry=model.task,task_embedding_hashes=model.task_embedding_hashes,
        source_hashes=model.source_hashes,initialization_audit=model.initialization_audit,
        epoch=epoch,gradient_clip=None,cpu_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,**metadata)
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
        parameter_names = {id(parameter):name for name,parameter in model.named_parameters()}
        payload['optimizer_parameter_names'] = [[parameter_names[id(p)] for p in group['params']]
                                                for group in optimizer.param_groups]
    if scheduler is not None:
        payload['scheduler'] = scheduler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def load_model_checkpoint(model,path,config):
    validate_config(config)
    payload = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('model_contract') != CONTRACT or payload.get('protocol') != config['protocol']:
        raise ValueError('Wrong architecture, variant, dataset, split or checkpoint identity')
    if payload.get('config') != config or payload.get('loss_protocol') != quality_loss_protocol(config):
        raise ValueError('Checkpoint configuration/loss mismatch')
    if payload.get('task_registry') != model.task or payload.get('task_embedding_hashes') != model.task_embedding_hashes:
        raise ValueError('Checkpoint task table/adapter mapping mismatch')
    if payload.get('source_hashes') != model.source_hashes or payload.get('gradient_clip','missing') is not None:
        raise ValueError('Checkpoint source hashes or gradient policy mismatch')
    phase = payload.get('fusion_phase')
    validate_phase(config,phase,payload['epoch'])
    expected, state = model.compact_state_dict(), payload['model']
    if state.keys() != expected.keys():
        raise ValueError('Checkpoint tensor coverage mismatch')
    for name, value in expected.items():
        saved = state[name]
        if value.shape != saved.shape or value.dtype != saved.dtype:
            raise ValueError('Checkpoint tensor shape/dtype mismatch: '+name)
        if name.startswith('dimension_encoder.') or name in ('fusion.position_encoding','blip_temporal.position_encoding'):
            if not torch.equal(value.cpu(),saved.cpu()):
                raise ValueError('Fixed table/position mismatch: '+name)
    missing, extra = model.load_state_dict(state,strict=False)
    if extra or any(not name.startswith(FROZEN) for name in missing):
        raise ValueError('Unexpected missing checkpoint tensors')
    model.configure_phase(phase)
    return payload
