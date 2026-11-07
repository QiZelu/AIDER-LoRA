"""Disposable real batch8 checks for every training phase and checkpoint reload."""
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset.joint import PairDataset
from training.engine.audit import digest, optimizer_snapshot
from training.engine.checkpoint import save_checkpoint, load_model_checkpoint
from training.engine.diagnostics import module_gradient_norms
from training.engine.gradients import prepare_gradients
from training.engine.io import write_json
from training.engine.losses import quality_training_loss
from training.engine.memory import check_memory
from training.engine.optim import build_optimizer, _scheduler
from training.engine.stages import switch_to_adapters, gate_state_digest
from training.engine.runtime import autocast_context, move_to_device
from training.engine.sampling import MixedPairSampler
from aider_modulation.qk_gate import gate_items
from aider_modulation.task_private import private_module_items
from training.protocol import validate_data


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
    for _,module in gate_items(model):
        if module.enabled != (phase != 'dense') or any(p.requires_grad != (phase != 'dense') for p in module.parameters()):
            raise AssertionError('Invalid task QK gate phase')
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
    # Two real batch8s; each leaves two private Heads inactive.
    return [[t*n+j for t in (0,1) for j in range(4)],
            [t*n+4+j for t in (2,3) for j in range(4)]]


def sampling_audit(config):
    sampler=MixedPairSampler(config['counts']['train_records'],config['loader']['sampler_seed'],8)
    before=torch.get_rng_state().clone();orders=[]
    for epoch in range(10):
        sampler.set_epoch(epoch);batches=sampler.batches()
        if len(batches)!=config['counts']['steps_per_epoch'] or any(len(b)!=8 for b in batches[:-1]) or not 1<=len(batches[-1])<=8:raise AssertionError('Batch count/size mismatch')
        if sorted(i for b in batches for i in b)!=list(range(config['counts']['train_records'])):raise AssertionError('Sampler coverage mismatch')
        orders.append(sampler.digest())
    if not torch.equal(before,torch.get_rng_state()):raise AssertionError('Sampler changed global RNG')
    return orders


def reload_probe(model,config,test,device,folder):
    ids = [task*test.video_count+task*2+j for task in range(4) for j in range(2)]
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
            model.fusion.blocks[0].fc1.weight.add_(0.2)
            for _,g in gate_items(model):
                g.linear.weight.add_(0.1)
                g.linear.bias.add_(0.3)
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
    orders = sampling_audit(config)
    initial = digest(model.compact_state_dict())
    phases = ['joint'] if config['training_strategy'] == 'joint' else ['dense','adapter']
    train, test = PairDataset(config,'train'), PairDataset(config,'val')
    if (train.video_count,test.video_count,len(train),len(test)) != tuple(config['counts'][k] for k in ('train_videos','test_videos','train_records','test_records')):
        raise AssertionError('Wrong original split')
    if len(list(private_module_items(model))) != 4 or any('.by_task.' in n for n,_ in model.named_parameters()):
        raise AssertionError('Only the four scoring Heads may be task-private')
    if len(list(gate_items(model))) != (0 if model.case=='no_qk' else 2) or any('moe' in name or 'experts' in name for name,_ in model.named_parameters()):
        raise AssertionError('Invalid dense Transformer/QK-gate architecture')
    device = next(model.parameters()).device
    optimizer = build_optimizer(model,config)
    scheduler = _scheduler(optimizer,config['counts']['total_updates'],.05,warmup_steps=config['training']['warmup_steps'])
    reports, deltas, transitions = [], {}, []
    for phase in phases:
        if phase == 'adapter':
            optimizer,scheduler,transition = switch_to_adapters(model,optimizer,scheduler,config,expected_updates=2)
            transitions.append(transition)
        check_precision(model,optimizer)
        frozen = frozen_digest(model)
        loader = DataLoader(train,batch_sampler=preflight_batches(train.video_count),num_workers=0,
                            generator=torch.Generator().manual_seed(9042))
        for step,cpu in enumerate(loader,1):
            model.train()
            batch = move_to_device(cpu,device)
            optimizer.zero_grad(set_to_none=True)
            previous = {name:optimizer_snapshot(module,optimizer) for name,_,module in private_items(model)}
            previous_base = fusion_base_digest(model)
            disabled_adapter_state = gate_state_digest(model) if phase=='dense' else None
            with autocast_context(device):
                output = model(batch)
            if output['routing']['concatenated_features'].shape != (8,8,model.fusion_width) or output['frame_scores'].shape != (8,8):
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
                if getattr(model,name) is None: continue
                if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in getattr(model,name).parameters()):
                    raise AssertionError('Active module has no gradient: '+name)
            if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.tv_backbone.model.text_encoder.parameters()):
                raise AssertionError('BLIP text gradient missing')
            if phase != 'adapter' and not any(p.grad is not None and torch.count_nonzero(p.grad) for _,p in model.fusion.base_parameters()):
                raise AssertionError('Fusion base gradient missing')
            if phase == 'adapter' and any(p.grad is not None for _,p in model.fusion.base_parameters()):
                raise AssertionError('Frozen fusion base received gradients')
            if phase != 'dense':
                for name,adapter in gate_items(model):
                    if adapter.linear.weight.grad is None or not torch.count_nonzero(adapter.linear.weight.grad):
                        raise AssertionError('QK channel gate has no effective gradient: '+name)
                    if any(p.grad is None for p in adapter.parameters()):
                        raise AssertionError('Shared adapter missed a gradient: '+name)
            if phase=='dense' and any(p.grad is not None or p in optimizer.state for _,a in gate_items(model) for p in a.parameters()):
                raise AssertionError('Disabled dense-stage QK gate has gradients or Adam state')
            for name in ('technical_projector','motion_projector','tv_projector','alpha_gate'):
                if getattr(model,name) is None: continue
                if any(p.grad is None for p in getattr(model,name).parameters()):
                    raise AssertionError('Shared FiLM/alpha gradient missing: '+name)
            gate_before = gate_state_digest(model)
            optimizer.step()
            if phase == 'adapter' and model.case!='no_qk' and gate_state_digest(model) == gate_before:
                raise AssertionError('QK gate did not update')
            scheduler.step()
            check_precision(model,optimizer)
            if disabled_adapter_state is not None and gate_state_digest(model)!=disabled_adapter_state:
                raise AssertionError('Dense-stage adapter weights changed')
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
                frozen_unchanged=True,fp32_state=True,disabled_gate_unchanged=phase=='dense',
                inactive_gate_adam_empty=phase=='dense',fusion_base_frozen=phase=='adapter'))
            print(f'Preflight {phase} batch {step}/2 passed, loss={float(loss.detach()):.6f}',flush=True)
            del output,loss
        model.zero_grad(set_to_none=True)
        deltas[phase] = reload_probe(model,config,test,device,folder)
    report = dict(passed=True,initial_state_sha256=initial,sampling_sha256=orders,steps=reports,
        phase_transitions=transitions,checkpoint_reload_max_abs_delta=max(deltas.values()),
        checkpoint_reload_by_phase=deltas,complete_blip_text_restored=True,
        full_shared_base_and_qk_gates_restored=True,peak_memory=watcher.report(),
        loss_protocol=config['loss_protocol'])
    write_json(folder / 'report.json',report)
    return initial,orders,report
