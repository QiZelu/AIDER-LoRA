"""Disposable real batch8 checks for every training phase and checkpoint reload."""
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.joint import PairDataset
from engine.audit import digest, optimizer_snapshot
from engine.checkpoint import save_checkpoint, load_model_checkpoint
from engine.diagnostics import module_gradient_norms
from engine.gradients import prepare_gradients
from engine.io import write_json
from engine.losses import quality_training_loss
from engine.memory import check_memory
from engine.optim import build_optimizer, _scheduler, transition_to_adapters
from engine.runtime import autocast_context, move_to_device
from engine.sampling import MixedPairSampler
from models.lora import lora_modules, task_adapter_items
from models.task_private import private_module_items
from utils.protocol import ROOT, validate_data


def check_precision(model,optimizer=None):
    if model.tv_backbone is not None and not all(p.requires_grad for p in model.tv_backbone.model.text_encoder.parameters()):
        raise AssertionError('Full BLIP text must remain trainable')
    for backbone in (model.vital_backbone,model.motion_backbone):
        if backbone is not None and any(p.requires_grad for p in backbone.parameters()):
            raise AssertionError('Visual backbone unfrozen')
    if model.tv_backbone is not None and any(p.requires_grad for p in model.tv_backbone.model.visual_encoder.parameters()):
        raise AssertionError('BLIP visual backbone unfrozen')
    phase = model.fusion.phase
    if any(p.requires_grad != (phase != 'adapter') for _,p in model.fusion.base_parameters()):
        raise AssertionError('Invalid fusion base freeze range')
    for _,module in lora_modules(model):
        if module.enabled != (phase != 'dense') or any(p.requires_grad != (phase != 'dense') for p in module.adapters.parameters()):
            raise AssertionError('Invalid task LoRA phase')
    if optimizer is not None:
        actual = [p for group in optimizer.param_groups for p in group['params']]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if len(actual) != len(expected) or {id(p) for p in actual} != expected:
            raise AssertionError('Optimizer trainable coverage mismatch')
    for name,parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            raise AssertionError('Non-FP32 trainable: '+name)
        if optimizer is not None:
            for key in ('exp_avg','exp_avg_sq'):
                value = optimizer.state.get(parameter,{}).get(key)
                if value is not None and value.dtype != torch.float32:
                    raise AssertionError('Non-FP32 Adam state: '+name)


def private_items(model):
    yield from private_module_items(model)
    yield from task_adapter_items(model)


def check_private_activity(model,ids):
    active = set(ids.detach().cpu().tolist())
    result = {}
    for name,task,module in private_items(model):
        expected = model.task_order.index(task) in active and any(p.requires_grad for p in module.parameters())
        if any((p.grad is not None) != expected for p in module.parameters()):
            raise AssertionError('Private gradient presence mismatch: '+name)
        result[name] = expected
    return result


def frozen_digest(model):
    state = {name:p for name,p in model.named_parameters() if not p.requires_grad}
    state.update(dict(model.named_buffers()))
    return digest(state)


def fusion_base_digest(model):
    return digest(dict(model.fusion.base_parameters()))


def preflight_batches(n):
    mixed = [[task*n+step*8+task*2+j for task in range(4) for j in range(2)] for step in range(2)]
    return mixed + [list(range(task*n+16,task*n+24)) for task in range(4)]


def sampling_audit():
    sampler = MixedPairSampler(1844,9042,8)
    before = torch.get_rng_state().clone()
    orders = []
    for epoch in range(10):
        sampler.set_epoch(epoch)
        batches = sampler.batches()
        if len(batches) != 231 or any(len(b) != 8 for b in batches[:-1]) or len(batches[-1]) != 4:
            raise AssertionError('Wrong batch/tail coverage')
        if sorted(i for b in batches for i in b) != list(range(1844)):
            raise AssertionError('Sampler missed/duplicated records')
        orders.append(sampler.digest())
    reference = json.loads((ROOT / 'reference/parent_preflight.json').read_text())['sampling_sha256']
    if orders != reference or not torch.equal(before,torch.get_rng_state()):
        raise AssertionError('Original CPU sample order/RNG changed')
    return orders


def reload_probe(model,config,test,device,folder):
    ids = [task*115+task*2+j for task in range(4) for j in range(2)]
    sample = move_to_device(next(iter(DataLoader(test,batch_sampler=[ids],num_workers=0,
        generator=torch.Generator().manual_seed(9042)))),device)
    model.eval()
    with tempfile.TemporaryDirectory(dir=folder) as temporary:
        path = Path(temporary) / 'preflight.pt'
        save_checkpoint(path,model,config,-1,preflight=True)
        expected_state = digest(model.compact_state_dict())
        expected_phase = model.fusion.phase
        with torch.no_grad(),autocast_context(device):
            expected = model(sample)['predictions'].float().clone()
        with torch.no_grad():
            model.fusion.blocks[0].fc1.base.weight.add_(0.2)
            next(iter(model.fusion.blocks[0].fc1.adapters.values())).B.weight.add_(0.1)
            next(model.fusion.heads.parameters()).add_(0.1)
            next(p for p in model.tv_backbone.model.text_encoder.parameters() if p.ndim == 1).add_(1.)
        payload = load_model_checkpoint(model,path,config)
        del payload
        if digest(model.compact_state_dict()) != expected_state or model.fusion.phase != expected_phase:
            raise AssertionError('Checkpoint did not restore the complete state/phase')
        with torch.no_grad(),autocast_context(device):
            actual = model(sample)['predictions'].float()
        delta = float((actual-expected).abs().max())
        if delta > 1e-6:
            raise AssertionError('Checkpoint reload error exceeds 1e-6')
    return delta


def preflight(model,config,folder,watcher):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    validate_data(config)
    orders = sampling_audit()
    initial = digest(model.compact_state_dict())
    phases = ['joint'] if config['training_strategy'] == 'joint' else ['dense','adapter']
    train, test = PairDataset(config,'train'), PairDataset(config,'val')
    if (train.video_count,test.video_count,len(train),len(test)) != (461,115,1844,460):
        raise AssertionError('Wrong original split')
    if len(list(lora_modules(model))) != 12 or any('moe' in name or 'experts' in name for name,_ in model.named_parameters()):
        raise AssertionError('Invalid dense Transformer/LoRA architecture')
    device = next(model.parameters()).device
    optimizer = build_optimizer(model,config)
    scheduler = _scheduler(optimizer,2310,.05,warmup_steps=116)
    reports, deltas, transitions = [], {}, []
    for phase in phases:
        if phase == 'adapter':
            optimizer,scheduler,transition = transition_to_adapters(model,optimizer,scheduler,config)
            transitions.append(transition)
        check_precision(model,optimizer)
        frozen = frozen_digest(model)
        loader = DataLoader(train,batch_sampler=preflight_batches(461),num_workers=0,
                            generator=torch.Generator().manual_seed(9042))
        for step,cpu in enumerate(loader,1):
            model.train()
            batch = move_to_device(cpu,device)
            optimizer.zero_grad(set_to_none=True)
            previous = {name:optimizer_snapshot(module,optimizer) for name,_,module in private_items(model)}
            previous_base = fusion_base_digest(model)
            with autocast_context(device):
                output = model(batch)
            if output['routing']['concatenated_features'].shape != (8,8,1536) or output['frame_scores'].shape != (8,8):
                raise AssertionError('Fusion shape changed')
            torch.testing.assert_close(output['predictions'],output['frame_scores'].mean(1),atol=0,rtol=0)
            loss,_,diagnostics = quality_training_loss(model,output,batch,config)
            if loss.dtype != torch.float32 or not torch.isfinite(loss):
                raise AssertionError('Expected finite FP32 loss')
            loss.backward()
            activity = check_private_activity(model,batch['task_id'])
            gradients = prepare_gradients(model,config,f'preflight {phase}/{step}')
            gradients['module_norms'] = module_gradient_norms(model)
            for name in ('technical_expert','blip_temporal'):
                if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in getattr(model,name).parameters()):
                    raise AssertionError('Active module has no gradient: '+name)
            if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.tv_backbone.model.text_encoder.parameters()):
                raise AssertionError('BLIP text gradient missing')
            if phase != 'adapter' and not any(p.grad is not None and torch.count_nonzero(p.grad) for _,p in model.fusion.base_parameters()):
                raise AssertionError('Fusion base gradient missing')
            if phase == 'adapter' and any(p.grad is not None for _,p in model.fusion.base_parameters()):
                raise AssertionError('Frozen fusion base received gradients')
            if phase != 'dense':
                for name,task,adapter in task_adapter_items(model):
                    if activity[name] and not torch.count_nonzero(adapter.B.weight.grad):
                        raise AssertionError('Active LoRA B has no effective gradient: '+name)
            optimizer.step()
            scheduler.step()
            check_precision(model,optimizer)
            for name,_,module in private_items(model):
                if not activity[name] and optimizer_snapshot(module,optimizer) != previous[name]:
                    raise AssertionError('Absent/disabled private parameter or Adam state changed: '+name)
            if phase == 'adapter' and fusion_base_digest(model) != previous_base:
                raise AssertionError('Stage-two fusion base changed')
            if frozen_digest(model) != frozen:
                raise AssertionError('Frozen weights/fixed buffers changed')
            check_memory(watcher.report())
            reports.append(dict(phase=phase,step=step,task_counts=torch.bincount(cpu['task_id'],minlength=4).tolist(),
                loss=diagnostics,gradients=gradients,absent_private_parameter_and_adam_unchanged=True,
                frozen_unchanged=True,fp32_state=True))
            print(f'Preflight {phase} batch {step}/6 passed, loss={float(loss.detach()):.6f}',flush=True)
            del output,loss
        model.zero_grad(set_to_none=True)
        deltas[phase] = reload_probe(model,config,test,device,folder)
    report = dict(passed=True,initial_state_sha256=initial,sampling_sha256=orders,steps=reports,
        phase_transitions=transitions,checkpoint_reload_max_abs_delta=max(deltas.values()),
        checkpoint_reload_by_phase=deltas,complete_blip_text_restored=True,
        full_shared_base_and_all_task_loras_restored=True,peak_memory=watcher.report(),
        loss_protocol=config['loss_protocol'])
    write_json(folder / 'report.json',report)
    return initial,orders,report
