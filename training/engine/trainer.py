"""Ten-epoch joint or 5+5 dense/QK gate experiments, fresh after real preflight."""
import csv
import gc
import json
import math
import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from training.dataset.loader import build_loader
from training.engine.audit import digest, file_digest, parameter_counts
from training.engine.checkpoint import load_model_checkpoint, save_checkpoint
from training.engine.diagnostics import module_gradient_norms, task_coefficients, gate_diagnostics
from training.engine.evaluation import evaluate, export, validate_metrics
from training.engine.gradients import gradient_statistics, prepare_gradients
from training.engine.io import write_csv, write_json
from training.engine.losses import quality_training_loss, TrainingLossSummary
from training.engine.memory import MemoryWatch, check_memory, initialize_process_memory
from training.engine.optim import _scheduler, build_optimizer
from training.engine.stages import switch_to_adapters, gate_state_digest
from training.engine.preflight import check_precision, check_private_activity, preflight, fusion_base_digest
from training.engine.runtime import autocast_context, move_to_device
from training.protocol import initialize_model
from training.protocol import (TASKS,load_prepared,read_rows,selected_source_hashes,
                            validate_data,phase_for_epoch,best_eligible)


def finetune_epoch(model,loader,optimizer,scheduler,device,epoch,config,watcher,root):
    loader.batch_sampler.set_epoch(epoch)
    indices, compositions, gradients, private_steps, loss_records = [], [], [], {}, []
    loss_summary = TrainingLossSummary()
    base_before = fusion_base_digest(model) if model.fusion.phase == 'adapter' else None
    adapter_before = gate_state_digest(model) if model.fusion.phase == 'dense' else None
    progress=tqdm(loader,desc=f'Epoch {epoch+1}/10 [{model.fusion.phase}]',unit='batch',mininterval=2,dynamic_ncols=True,ascii=True)
    for step,cpu in enumerate(progress,1):
        model.train()
        batch = move_to_device(cpu,device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            output = model(batch)
        loss, values, diagnostics = quality_training_loss(model,output,batch,config)
        loss.backward()
        active = check_private_activity(model,batch['task_id'])
        for name,present in active.items():
            private_steps[name] = private_steps.get(name,0) + int(present)
        if model.fusion.phase == 'adapter' and any(p.grad is not None for _,p in model.fusion.base_parameters()):
            raise AssertionError('Frozen fusion base received gradients')
        grad = prepare_gradients(model,config,f'epoch={epoch+1} step={step}')
        grad.update(step=step,epoch=epoch+1,module_norms=module_gradient_norms(model))
        optimizer.step()
        scheduler.step()
        check_memory(watcher.report())
        update = epoch*config['counts']['steps_per_epoch'] + step
        loss_summary.add(values,diagnostics)
        loss_records.append(dict(epoch=epoch+1,step=step,update=update,
            phase='before_optimizer_step',fusion_phase=model.fusion.phase,**diagnostics))
        indices.append(cpu['pair_index'].tolist())
        compositions.append(torch.bincount(cpu['task_id'],minlength=4).tolist())
        gradients.append(grad)
        progress.set_postfix(loss=f'{float(loss.detach()):.6f}',lr=f'{scheduler.get_last_lr()[0]:.2e}',refresh=False)
        if step == 1 or step % 20 == 0:
            write_json(root / 'status.json',dict(stage='training',variant=config['training_strategy'],
                fusion_phase=model.fusion.phase,epoch=epoch+1,step=step,completed_epochs=epoch,
                optimizer_updates=update,physical_gpu=config['experiment']['physical_gpu']))
        del output,loss,values
    if indices != loader.batch_sampler.batches():
        raise AssertionError('Actual sample order differs from original CPU sampler')
    if base_before is not None and fusion_base_digest(model) != base_before:
        raise AssertionError('Frozen fusion base changed during epoch')
    if adapter_before is not None and gate_state_digest(model) != adapter_before:
        raise AssertionError('Disabled QK gate changed during dense epoch')
    check_precision(model,optimizer)
    summary = loss_summary.report()
    return dict(train_loss=summary['regression_loss_sample_mean'],loss_summary=summary,loss_records=loss_records,
        pair_indices=indices,batch_task_counts=compositions,order_sha256=loader.batch_sampler.digest(),
        gradient_records=gradients,private_update_steps=private_steps,fusion_base_sha256=base_before,
        disabled_gate_sha256=adapter_before)


def select_best(root,model,config,epoch,metrics,best):
    validate_metrics(metrics)
    score = metrics['macro']['srocc']
    if best_eligible(config,epoch) and score > best:
        save_checkpoint(root / 'best.pt',model,config,epoch,score=score,
                        selection='test_macro_srocc_earliest_strict_improvement')
        return score
    return best


def train(config,mode='train'):
    root = Path(config['protocol']['output_root'])
    load_prepared(root)
    if mode == 'train':
        if any((root / name).exists() for name in ('best.pt','last.pt','history.json','preflight','.training_claim')):
            raise FileExistsError('Existing shared FiLM/QK-gated run; refuse overwrite or automatic retry')
        with (root / '.training_claim').open('x') as handle:
            handle.write(str(os.getpid())+'\n')
        run = root
    else:
        run = root / 'preflight_only'
        run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    watcher = None
    try:
        write_json(run / 'status.json',dict(stage='initializing',optimizer_updates=0))
        initialize_process_memory()
        torch.cuda.set_per_process_memory_fraction(14*1024**3/torch.cuda.get_device_properties(0).total_memory)
        with MemoryWatch() as watcher:
            sources = selected_source_hashes(config)
            model = initialize_model(config).cuda()
            model.source_hashes = sources
            cpu_rng = model.construction_rng_sha256
            cuda_rng = digest({'cuda_rng':torch.cuda.get_rng_state(0)})
            model.initialization_audit['cuda_rng_sha256'] = cuda_rng
            for name,data in (('sources.json',sources),('initialization_audit.json',model.initialization_audit),
                              ('parameters.json',parameter_counts(model)),('loss_protocol.json',config['loss_protocol'])):
                write_json(run / name,data)
            write_json(run / 'initial_film.json',task_coefficients(model))
            write_json(run / 'initial_qk_gate.json',gate_diagnostics(model))
            write_json(run / 'status.json',dict(stage='preflight',variant=config['training_strategy'],optimizer_updates=0))
            expected,orders,report = preflight(model,config,run / 'preflight',watcher)
            write_json(run / 'preflight.json',report)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if mode == 'preflight':
                write_json(run / 'status.json',dict(stage='preflight_complete',formal_training_started=False))
                return
            write_json(root / 'status.json',dict(stage='reinitializing_after_preflight',optimizer_updates=0))
            model = initialize_model(config).cuda()
            model.source_hashes = sources
            if (digest(model.compact_state_dict()) != expected or model.construction_rng_sha256 != cpu_rng
                    or digest({'cuda_rng':torch.cuda.get_rng_state(0)}) != cuda_rng):
                raise AssertionError('Preflight updates/RNG entered formal training')
            write_json(root / 'fresh_initialization_audit.json',dict(initial_state_sha256=expected,
                cpu_rng_sha256=cpu_rng,cuda_rng_sha256=cuda_rng,parameters_equal=True,cpu_rng_equal=True,
                cuda_rng_equal=True,preflight_updates_discarded=True,
                shared_transformer_base_equal_parent=model.initialization_audit['shared_transformer_base_equal_parent']))
            train_loader, test_loader = build_loader(config,'train'), build_loader(config,'val')
            if len(train_loader) != config['counts']['steps_per_epoch'] or len(test_loader.dataset) != config['counts']['test_records']:
                raise AssertionError('Wrong original loaders')
            optimizer = build_optimizer(model,config)
            scheduler = _scheduler(optimizer,config['counts']['total_updates'],.05,warmup_steps=config['training']['warmup_steps'])
            best, history, device = -float('inf'), [], torch.device('cuda:0')
            for epoch in range(10):
                if phase_for_epoch(config,epoch) == 'adapter' and model.fusion.phase == 'dense':
                    optimizer,scheduler,transition = switch_to_adapters(model,optimizer,scheduler,config,expected_updates=config['counts']['transition_updates'])
                    if transition['optimizer_updates'] != config['counts']['transition_updates']:
                        raise AssertionError('Wrong phase transition update')
                    transition['stage1_checkpoint_sha256'] = file_digest(root / 'stage1_last.pt')
                    transition['fusion_base_sha256'] = fusion_base_digest(model)
                    write_json(root / 'phase_transition.json',transition)
                    write_json(root / 'stage2_parameters.json',parameter_counts(model))
                if model.fusion.phase != phase_for_epoch(config,epoch):
                    raise AssertionError('Wrong formal training phase')
                write_json(root / 'status.json',dict(stage='training',variant=config['training_strategy'],
                    fusion_phase=model.fusion.phase,epoch=epoch+1,completed_epochs=epoch,optimizer_updates=epoch*config['counts']['steps_per_epoch'],
                    physical_gpu=config['experiment']['physical_gpu']))
                training = finetune_epoch(model,train_loader,optimizer,scheduler,device,epoch,config,watcher,root)
                if training['order_sha256'] != orders[epoch]:
                    raise AssertionError('Formal sampler order changed')
                save_checkpoint(root / 'last.pt',model,config,epoch,optimizer,scheduler,
                    optimizer_updates=(epoch+1)*config['counts']['steps_per_epoch'],stage='epoch_training_complete')
                if config['training_strategy'] == 'two_stage' and epoch == 4:
                    # Later last.pt writes use atomic replacement; the hard link
                    # therefore keeps an immutable complete epoch-five snapshot.
                    os.link(root / 'last.pt',root / 'stage1_last.pt')
                write_json(root / 'status.json',dict(stage='testing',variant=config['training_strategy'],
                    fusion_phase=model.fusion.phase,epoch=epoch+1,optimizer_updates=(epoch+1)*config['counts']['steps_per_epoch']))
                rows,metrics,diagnostics = evaluate(model,test_loader,device)
                validate_metrics(metrics)
                export(root / 'epochs',f'epoch_{epoch+1:02d}',rows,metrics,diagnostics,epoch,root / 'last.pt')
                write_json(root / 'qk_gate' / f'epoch_{epoch+1:02d}.json',gate_diagnostics(model))
                write_json(root / 'film' / f'epoch_{epoch+1:02d}_task_coefficients.json',task_coefficients(model))
                write_json(root / 'sampling' / f'epoch_{epoch+1:02d}.json',
                    {key:training[key] for key in ('pair_indices','batch_task_counts','order_sha256')})
                write_json(root / 'gradients' / f'epoch_{epoch+1:02d}.json',dict(
                    summary=gradient_statistics(training['gradient_records']),steps=training['gradient_records'],
                    private_update_steps=training['private_update_steps']))
                write_json(root / 'losses' / f'epoch_{epoch+1:02d}.json',dict(
                    summary=training['loss_summary'],steps=training['loss_records']))
                history.append(dict(epoch=epoch+1,fusion_phase=model.fusion.phase,train_loss=training['train_loss'],
                    train_loss_semantics='regression_loss_sample_mean',losses=training['loss_summary'],
                    optimizer_updates=(epoch+1)*config['counts']['steps_per_epoch'],fusion_base_sha256=training['fusion_base_sha256'],disabled_gate_sha256=training['disabled_gate_sha256'],test=dict(metrics=metrics)))
                write_json(root / 'history.json',history)
                best = select_best(root,model,config,epoch,metrics,best)
                write_json(root / 'process_memory.json',watcher.report())
                print(f"Epoch {epoch+1}/10 complete | macro SROCC={metrics['macro']['srocc']:.6f} | best={best if math.isfinite(best) else 'not eligible yet'}",flush=True)
            if scheduler.last_epoch != config['counts']['total_updates']:
                raise AssertionError('Incomplete LR schedule')
            del optimizer,scheduler
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            write_json(root / 'status.json',dict(stage='evaluating_best_last',completed_epochs=10,optimizer_updates=config['counts']['total_updates']))
            reload_checks = {}
            for kind in ('best','last'):
                checkpoint = root / f'{kind}.pt'
                payload = load_model_checkpoint(model,checkpoint,config)
                epoch = payload['epoch']
                del payload
                rows,metrics,diagnostics = evaluate(model,test_loader,device)
                original_report = json.loads((root / 'epochs' / f'epoch_{epoch+1:02d}_evaluation.json').read_text())
                original_rows = {(r['video_id'],r['task']):r['prediction'] for r in original_report['samples']}
                if len(original_rows) != config['counts']['test_records']:
                    raise ValueError('Incomplete reference epoch predictions')
                reload_error = max(abs(row['prediction']-original_rows[row['video_id'],row['task']]) for row in rows)
                if reload_error > 1e-6:
                    raise ValueError('Best/last full-test checkpoint reload error exceeds 1e-6')
                reload_checks[kind] = dict(epoch=epoch+1,predictions=len(rows),max_abs_error=reload_error)
                export(root,kind,rows,metrics,diagnostics,epoch,checkpoint)
                write_json(root / f'{kind}_qk_gate.json',gate_diagnostics(model))
                write_json(root / f'{kind}_task_coefficients.json',task_coefficients(model))
                check_memory(watcher.report())
            write_json(root / 'checkpoint_reload_verification.json',dict(passed=True,checkpoints=reload_checks))
            write_json(root / 'process_memory.json',watcher.report())
            summarize_run(root)
            write_json(root / 'status.json',dict(stage='complete',variant=config['training_strategy'],
                completed_epochs=10,optimizer_updates=config['counts']['total_updates']))
    except Exception as error:
        write_json(run / 'status.json',dict(stage='failed',variant=config['training_strategy'],error=repr(error),
            automatic_retry=False,last_checkpoint_saved=(root / 'last.pt').exists()))
        if watcher is not None:
            write_json(run / 'process_memory.json',watcher.report())
        raise


def summarize_run(root):
    root=Path(root);cfg=load_prepared(root)
    history=json.loads((root/'history.json').read_text())
    if len(history)!=10:raise ValueError('Incomplete epoch history')
    best=max(history[5:],key=lambda r:r['test']['metrics']['macro']['srocc'])['epoch']
    for i,r in enumerate(history):
        if r['epoch']!=i+1 or r['optimizer_updates']!=(i+1)*cfg['counts']['steps_per_epoch'] or r['fusion_phase']!=phase_for_epoch(cfg,i):
            raise ValueError('Incomplete phase/update history')
        loss=json.loads((root/'losses'/f'epoch_{i+1:02d}.json').read_text())
        if len(loss['steps'])!=cfg['counts']['steps_per_epoch'] or loss['summary']['samples']!=cfg['counts']['train_records']:raise ValueError('Incomplete loss/sample records')
        validate_metrics(r['test']['metrics'])
    if len({r['fusion_base_sha256'] for r in history[5:]})!=1:raise ValueError('Stage-two Fusion changed')
    if len({r['disabled_gate_sha256'] for r in history[:5]})!=1:raise ValueError('Inactive gate changed')
    transition=json.loads((root/'phase_transition.json').read_text())
    if transition['optimizer_updates']!=cfg['counts']['transition_updates'] or transition['lr_before']!=transition['lr_after']:
        raise ValueError('Invalid phase transition')
    preflight=json.loads((root/'preflight.json').read_text())
    fresh=json.loads((root/'fresh_initialization_audit.json').read_text())
    reload=json.loads((root/'checkpoint_reload_verification.json').read_text())
    if not preflight['passed'] or len(preflight['steps'])!=4 or preflight['checkpoint_reload_max_abs_delta']>1e-6:
        raise ValueError('Invalid preflight')
    if not all(fresh[k] for k in ('parameters_equal','cpu_rng_equal','cuda_rng_equal','preflight_updates_discarded')):
        raise ValueError('Initialization not restored')
    if not reload['passed'] or any(x['max_abs_error']>1e-6 or x['predictions']!=cfg['counts']['test_records'] for x in reload['checkpoints'].values()):
        raise ValueError('Invalid full-test reload')
    expected={(r['id'],t):r['labels'][t] for r in read_rows(cfg['data']['val_manifest']) for t in TASKS}
    rows=[]
    for kind,epoch in [('last',10),('best',best)]:
        report=json.loads((root/f'{kind}_evaluation.json').read_text())
        if report['checkpoint_epoch']!=epoch or report['checkpoint_sha256']!=file_digest(root/f'{kind}.pt'):
            raise ValueError('Wrong checkpoint identity')
        actual={(r['video_id'],r['task']):r['target'] for r in report['samples']}
        if len(report['samples'])!=cfg['counts']['test_records'] or actual.keys()!=expected.keys() or any(abs(actual[k]-expected[k])>1e-6 for k in expected):
            raise ValueError('Wrong prediction coverage or labels')
        validate_metrics(report['metrics'])
        for task in (*TASKS,'macro'):
            for metric in ('plcc','srocc','krocc','rmse'):
                if abs(report['metrics'][task][metric]-history[epoch-1]['test']['metrics'][task][metric])>1e-6:
                    raise ValueError('Reload metrics differ')
            rows.append(dict(checkpoint=kind,epoch=epoch,task=task,**{k:report['metrics'][task][k] for k in ('plcc','srocc','krocc','rmse')}))
        for suffix,count in [('predictions_long.csv',cfg['counts']['test_records']),('predictions_wide.csv',cfg['counts']['test_videos'])]:
            with (root/f'{kind}_{suffix}').open() as f:
                if sum(1 for _ in csv.DictReader(f))!=count:raise ValueError('Invalid CSV coverage')
    write_csv(root/'metrics.csv',rows)
    report=dict(complete=True,best_epoch=best,optimizer_updates=cfg['counts']['total_updates'],case='full',
        split_seed=cfg['data'].get('split_seed'),metrics=rows,data_audit=validate_data(cfg))
    write_json(root/'summary.json',report)
    return report
