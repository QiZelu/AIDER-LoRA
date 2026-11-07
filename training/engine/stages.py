"""Audited dense-to-adapter transition; keep the original AdamW/LR implementation."""
import torch

from training.engine.audit import digest
from training.engine.optim import transition_to_adapters
from aider_modulation.qk_gate import gate_items


def gate_state_digest(model):
    return digest({name+'.'+key:value for name,adapter in gate_items(model)
                   for key,value in adapter.state_dict().items()})


def switch_to_adapters(model,optimizer,scheduler,config,expected_updates=None):
    if expected_updates is None:expected_updates=config['counts']['transition_updates']
    if model.fusion.phase!='dense' or scheduler.last_epoch!=expected_updates:
        raise ValueError('Dense-to-adapter transition at the wrong phase/update')
    adapters=list(gate_items(model))
    adapter_ids={id(p) for _,a in adapters for p in a.parameters()}
    base_ids={id(p) for _,p in model.fusion.base_parameters()}
    names={id(p):name for name,p in model.named_parameters()}
    before_flags={id(p):p.requires_grad for p in model.parameters()}
    for name,adapter in adapters:
        if any(p.requires_grad or p.grad is not None or p in optimizer.state for p in adapter.parameters()):
            raise AssertionError('Disabled dense-stage adapter acquired gradients/Adam state: '+name)
        if adapter.state_hash() != adapter.initial_state_hash:
            raise AssertionError('Disabled adapter changed before stage two: '+name)
    continuing=[p for p in model.parameters() if p.requires_grad and id(p) not in base_ids]
    before={id(p):dict(optimizer.state.get(p,{})) for p in continuing}
    versions={(id(p),key):value._version for p in continuing for key,value in before[id(p)].items()
              if isinstance(value,torch.Tensor)}
    cpu_rng=torch.get_rng_state().clone()
    device=next(model.parameters()).device
    cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    adapter_digest=gate_state_digest(model)
    optimizer,scheduler,report=transition_to_adapters(model,optimizer,scheduler,config)
    if scheduler.last_epoch!=expected_updates or report['lr_before']!=report['lr_after']:
        raise AssertionError('Stage transition reset LR or scheduler')
    for p in model.parameters():
        expected=True if id(p) in adapter_ids else (False if id(p) in base_ids else before_flags[id(p)])
        if p.requires_grad!=expected:
            raise AssertionError('Unexpected stage-two freeze range: '+names[id(p)])
    for p in continuing:
        after=optimizer.state.get(p,{})
        if before[id(p)].keys()!=after.keys():
            raise AssertionError('Continuing parameter lost Adam state: '+names[id(p)])
        for key,value in before[id(p)].items():
            if isinstance(value,torch.Tensor):
                if after[key] is not value or after[key]._version!=versions[(id(p),key)]:
                    raise AssertionError('Continuing Adam tensor replaced/mutated: '+names[id(p)])
            elif after[key]!=value:
                raise AssertionError('Continuing Adam scalar changed: '+names[id(p)])
    if any(p in optimizer.state for _,a in adapters for p in a.parameters()):
        raise AssertionError('New adapter Adam state must be empty')
    if gate_state_digest(model)!=adapter_digest or not torch.equal(cpu_rng,torch.get_rng_state()):
        raise AssertionError('Transition changed initial adapter values or CPU RNG')
    if cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state(device)):
        raise AssertionError('Transition consumed CUDA RNG')
    report.update(continuing_adam_tensor_identity_and_versions_preserved=True,
        continuing_parameters=len(continuing),other_module_trainability_unchanged=True,
        adapter_values_preserved=True,new_adapter_adam_empty=True,cpu_rng_preserved=True,
        cuda_rng_preserved=cuda_rng is not None,
        base_parameters_removed_from_optimizer=True,expected_transition_updates=expected_updates)
    return optimizer,scheduler,report
