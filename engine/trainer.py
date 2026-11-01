"""Ten-epoch joint or 5+5 dense/LoRA experiments, fresh after real preflight."""
import csv
import gc
import json
import math
import os
from pathlib import Path

import cv2
import torch

from dataset.loader import build_loader
from engine.audit import digest, file_digest, parameter_counts
from engine.checkpoint import load_model_checkpoint, save_checkpoint
from engine.diagnostics import module_gradient_norms, task_coefficients, lora_diagnostics
from engine.evaluation import evaluate, export, validate_metrics
from engine.gradients import gradient_statistics, prepare_gradients
from engine.io import write_csv, write_json
from engine.losses import quality_training_loss, TrainingLossSummary
from engine.memory import MemoryWatch, check_memory, initialize_process_memory
from engine.optim import _scheduler, build_optimizer
from engine.stages import switch_to_adapters, lora_state_digest
from engine.preflight import check_precision, check_private_activity, preflight, fusion_base_digest
from engine.runtime import autocast_context, move_to_device
from utils.initialization import initialize_model
from utils.protocol import (ROOT,TASKS,SOURCE,CONTROL,load_prepared,read_rows,selected_source_hashes,
                            validate_data,phase_for_epoch,best_eligible,validate_reference)


def finetune_epoch(model,loader,optimizer,scheduler,device,epoch,config,watcher,root):
    loader.batch_sampler.set_epoch(epoch)
    indices, compositions, gradients, private_steps, loss_records = [], [], [], {}, []
    loss_summary = TrainingLossSummary()
    base_before = fusion_base_digest(model) if model.fusion.phase == 'adapter' else None
    adapter_before = lora_state_digest(model) if model.fusion.phase == 'dense' else None
    for step,cpu in enumerate(loader,1):
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
        update = epoch*231 + step
        loss_summary.add(values,diagnostics)
        loss_records.append(dict(epoch=epoch+1,step=step,update=update,
            phase='before_optimizer_step',fusion_phase=model.fusion.phase,**diagnostics))
        indices.append(cpu['pair_index'].tolist())
        compositions.append(torch.bincount(cpu['task_id'],minlength=4).tolist())
        gradients.append(grad)
        print(json.dumps(dict(epoch=epoch+1,step=step,update=update,fusion_phase=model.fusion.phase,
            loss=float(loss.detach()),global_grad_norm=grad['global_norm_fp64'],
            learning_rates=scheduler.get_last_lr())),flush=True)
        if step == 1 or step % 20 == 0:
            write_json(root / 'status.json',dict(stage='training',variant=config['training_strategy'],
                fusion_phase=model.fusion.phase,epoch=epoch+1,step=step,completed_epochs=epoch,
                optimizer_updates=update,physical_gpu=config['experiment']['physical_gpu']))
        del output,loss,values
    if indices != loader.batch_sampler.batches():
        raise AssertionError('Actual sample order differs from original CPU sampler')
    if base_before is not None and fusion_base_digest(model) != base_before:
        raise AssertionError('Frozen fusion base changed during epoch')
    if adapter_before is not None and lora_state_digest(model) != adapter_before:
        raise AssertionError('Disabled LoRA changed during dense epoch')
    check_precision(model,optimizer)
    summary = loss_summary.report()
    return dict(train_loss=summary['regression_loss_sample_mean'],loss_summary=summary,loss_records=loss_records,
        pair_indices=indices,batch_task_counts=compositions,order_sha256=loader.batch_sampler.digest(),
        gradient_records=gradients,private_update_steps=private_steps,fusion_base_sha256=base_before,
        disabled_lora_sha256=adapter_before)


def select_best(root,model,config,epoch,metrics,best):
    validate_metrics(metrics)
    score = metrics['macro']['srocc']
    if best_eligible(config,epoch) and score > best:
        save_checkpoint(root / 'best.pt',model,config,epoch,score=score,
                        selection='test_macro_srocc_earliest_strict_improvement')
        return score
    return best


def summarize(root):
    root = Path(root)
    config = load_prepared(root)
    history = json.loads((root / 'history.json').read_text())
    if len(history) != 10:
        raise ValueError('Incomplete ten-epoch history')
    candidates = [row for row in history if best_eligible(config,row['epoch']-1)]
    best = max(candidates,key=lambda row:row['test']['metrics']['macro']['srocc'])['epoch']
    for index,row in enumerate(history):
        if (row['epoch'] != index+1 or row['optimizer_updates'] != (index+1)*231
                or row['fusion_phase'] != phase_for_epoch(config,index)):
            raise ValueError('Epoch/update/phase mismatch')
        validate_metrics(row['test']['metrics'])
        losses = json.loads((root / 'losses' / f'epoch_{index+1:02d}.json').read_text())
        if len(losses['steps']) != 231 or losses['summary'] != row['losses'] or row['losses']['samples'] != 1844:
            raise ValueError('Loss sample/update coverage mismatch')
        for step,record in enumerate(losses['steps'],1):
            if (record['update'] != index*231+step or not math.isfinite(record['total_loss'])
                    or record['total_loss'] != record['regression_loss']):
                raise ValueError('Incorrect regression-only loss record')
    expected = {(row['id'],task):row['labels'][task] for row in read_rows(config['data']['val_manifest']) for task in TASKS}
    metrics = []
    for kind,epoch in (('best',best),('last',10)):
        report = json.loads((root / f'{kind}_evaluation.json').read_text())
        samples = report['samples']
        actual = {(row['video_id'],row['task']):row['target'] for row in samples}
        if (not report['completed'] or report['checkpoint_epoch'] != epoch or len(samples) != 460
                or actual.keys() != expected.keys() or any(abs(actual[k]-expected[k]) > 1e-6 for k in expected)
                or report['checkpoint_sha256'] != file_digest(root / f'{kind}.pt')):
            raise ValueError('Best/last checkpoint identity or prediction coverage mismatch')
        validate_metrics(report['metrics'])
        for task in (*TASKS,'macro'):
            for key in ('plcc','srocc','krocc','rmse'):
                if not math.isclose(report['metrics'][task][key],history[epoch-1]['test']['metrics'][task][key],abs_tol=1e-6,rel_tol=0):
                    raise ValueError('Reloaded metrics differ from selected epoch')
            metrics.append(dict(checkpoint=kind,epoch=epoch,task=task,
                **{key:report['metrics'][task][key] for key in ('plcc','srocc','krocc','rmse')}))
        for suffix,count in (('predictions_long.csv',460),('predictions_wide.csv',115)):
            with (root / f'{kind}_{suffix}').open() as handle:
                if sum(1 for _ in csv.DictReader(handle)) != count:
                    raise ValueError('Incomplete prediction CSV')
        if len(json.loads((root / f'{kind}_film_diagnostics.json').read_text())) != 460:
            raise ValueError('Incomplete FiLM diagnostics')
        lora = json.loads((root / f'{kind}_lora.json').read_text())
        if len(lora['projections']) != 12 or not lora['adapters_enabled']:
            raise ValueError('Incomplete task LoRA diagnostics')
    prior = json.loads((root / 'preparation/prior_audit.json').read_text())
    preflight_report = json.loads((root / 'preflight.json').read_text())
    fresh = json.loads((root / 'fresh_initialization_audit.json').read_text())
    paired = json.loads((root / 'preparation/initialization_comparison/paired_initialization_audit.json').read_text())
    reloaded = json.loads((root / 'checkpoint_reload_verification.json').read_text())
    if (not paired.get('passed') or not reloaded.get('passed')
            or set(reloaded['checkpoints']) != {'best','last'}
            or any(r['max_abs_error'] > 1e-6 or r['predictions'] != 460 for r in reloaded['checkpoints'].values())):
        raise ValueError('Paired initialization or full-test checkpoint reload verification incomplete')
    expected_steps = 6 if config['training_strategy'] == 'joint' else 12
    if (not prior['all_sha256_verified'] or prior['videos'] != 576 or not preflight_report['passed']
            or len(preflight_report['steps']) != expected_steps or preflight_report['checkpoint_reload_max_abs_delta'] > 1e-6
            or not all(fresh[key] for key in ('parameters_equal','cpu_rng_equal','cuda_rng_equal','preflight_updates_discarded'))):
        raise ValueError('Preflight/fresh initialization/cache audit incomplete')
    if config['training_strategy'] == 'two_stage':
        transition = json.loads((root / 'phase_transition.json').read_text())
        if (transition['optimizer_updates'] != 1155 or transition['lr_before'] != transition['lr_after']
                or not (root / 'stage1_last.pt').exists()
                or len({row['fusion_base_sha256'] for row in history[5:]}) != 1
                or len({row['disabled_lora_sha256'] for row in history[:5]}) != 1
                or not all(transition.get(key) for key in ('continuing_adam_tensor_identity_and_versions_preserved',
                    'other_module_trainability_unchanged','adapter_values_preserved','new_adapter_adam_empty'))):
            raise ValueError('Two-stage continuity/frozen-base audit incomplete')
    write_csv(root / 'metrics.csv',metrics)
    write_csv(root / 'loss_history.csv',[dict(epoch=row['epoch'],fusion_phase=row['fusion_phase'],**row['losses']) for row in history])
    report = dict(complete=True,variant=config['training_strategy'],best_epoch=best,optimizer_updates=2310,
        best_candidate_epochs=config['evaluation']['best_epochs'],metrics=metrics,loss_protocol=config['loss_protocol'],
        shared_lora_parameters=294912,channel_modulator_parameters=73824,shared_lora_and_gate_parameters=368736,
        sharing_policy=config['sharing_policy'],
        data_audit=validate_data(config),caveats=[
            'Test split selects best and reports final metrics; no separate validation set.',
            'Single model seed42; no automatic Baseline replacement.',
            'Dense epochs1-5; adapter epochs6-10; best selection only covers epochs6-10.',
            'The matched control uses the same raw CLIP embeddings and architecture with joint epochs1-10.',
            'All tasks reweight eight shared low-rank directions; differences are learned, not enforced.'])
    report['control_comparison'] = compare_control(root)
    write_json(root / 'summary.json',report)
    return report


def compare_control(root):
    validate_reference(check_parent_code=True)
    root=Path(root)
    rows=[]
    provenance={}
    for name,folder in (('joint',CONTROL),('two_stage',root)):
        provenance[name]=dict(path=str(folder),config_sha256=file_digest(folder/'config.yaml'),
            sources_sha256=file_digest(folder/'sources.json'),best_epochs=[1,10] if name=='joint' else [6,10])
    for kind in ('best','last'):
        control=json.loads((CONTROL/f'{kind}_evaluation.json').read_text())
        current=json.loads((root/f'{kind}_evaluation.json').read_text())
        for folder,report in ((CONTROL,control),(root,current)):
            if (not report['completed'] or len(report['samples'])!=460
                    or report['checkpoint_sha256']!=file_digest(folder/f'{kind}.pt')):
                raise ValueError('Comparison checkpoint identity/coverage mismatch')
        labels={(r['video_id'],r['task']):r['target'] for r in control['samples']}
        now={(r['video_id'],r['task']):r['target'] for r in current['samples']}
        if labels.keys()!=now.keys() or len(labels)!=460 or any(abs(labels[k]-now[k])>1e-6 for k in labels):
            raise ValueError('Joint/two-stage comparison labels differ')
        validate_metrics(control['metrics']);validate_metrics(current['metrics'])
        for task in (*TASKS,'macro'):
            for metric in ('plcc','srocc','krocc','rmse'):
                before,after=control['metrics'][task][metric],current['metrics'][task][metric]
                rows.append(dict(checkpoint=kind,task=task,metric=metric,joint=before,two_stage=after,
                    difference=after-before,joint_epoch=control['checkpoint_epoch'],two_stage_epoch=current['checkpoint_epoch']))
    write_csv(root/'comparison_joint_vs_two_stage_metrics.csv',rows)
    report=dict(available=True,control=str(CONTROL),provenance=provenance,rows=rows,
        changes=['Dense epochs1-5 disable LoRA; adapter epochs6-10 freeze Fusion base only',
                 'FiLM, alpha, video summaries, private Heads and original non-Fusion trainable modules continue'],
        baseline_replaced=False,caveats=['Test split selects best; no separate validation split.',
            'Joint best candidates: epochs1-10; two-stage best candidates: epochs6-10.',
            'Both last checkpoints are epoch10.'])
    write_json(root/'comparison_joint_vs_two_stage.json',report)
    return dict(available=True,control=str(CONTROL),metrics_csv='comparison_joint_vs_two_stage_metrics.csv',
                caveat='Only training schedule/freezing changes; best candidate epochs differ.')


def train(config,mode='train'):
    root = Path(config['protocol']['output_root'])
    load_prepared(root)
    if mode == 'train':
        if any((root / name).exists() for name in ('best.pt','last.pt','history.json','preflight','.training_claim')):
            raise FileExistsError('Existing shared FiLM/conditional-LoRA run; refuse overwrite or automatic retry')
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
            write_json(run / 'initial_lora.json',lora_diagnostics(model))
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
            if len(train_loader) != 231 or len(test_loader.dataset) != 460:
                raise AssertionError('Wrong original loaders')
            optimizer = build_optimizer(model,config)
            scheduler = _scheduler(optimizer,2310,.05,warmup_steps=116)
            best, history, device = -float('inf'), [], torch.device('cuda:0')
            for epoch in range(10):
                if phase_for_epoch(config,epoch) == 'adapter' and model.fusion.phase == 'dense':
                    optimizer,scheduler,transition = switch_to_adapters(model,optimizer,scheduler,config,expected_updates=1155)
                    if transition['optimizer_updates'] != 1155:
                        raise AssertionError('Wrong phase transition update')
                    transition['stage1_checkpoint_sha256'] = file_digest(root / 'stage1_last.pt')
                    transition['fusion_base_sha256'] = fusion_base_digest(model)
                    write_json(root / 'phase_transition.json',transition)
                    write_json(root / 'stage2_parameters.json',parameter_counts(model))
                if model.fusion.phase != phase_for_epoch(config,epoch):
                    raise AssertionError('Wrong formal training phase')
                write_json(root / 'status.json',dict(stage='training',variant=config['training_strategy'],
                    fusion_phase=model.fusion.phase,epoch=epoch+1,completed_epochs=epoch,optimizer_updates=epoch*231,
                    physical_gpu=config['experiment']['physical_gpu']))
                training = finetune_epoch(model,train_loader,optimizer,scheduler,device,epoch,config,watcher,root)
                if training['order_sha256'] != orders[epoch]:
                    raise AssertionError('Formal sampler order changed')
                save_checkpoint(root / 'last.pt',model,config,epoch,optimizer,scheduler,
                    optimizer_updates=(epoch+1)*231,stage='epoch_training_complete')
                if config['training_strategy'] == 'two_stage' and epoch == 4:
                    # Later last.pt writes use atomic replacement; the hard link
                    # therefore keeps an immutable complete epoch-five snapshot.
                    os.link(root / 'last.pt',root / 'stage1_last.pt')
                write_json(root / 'status.json',dict(stage='testing',variant=config['training_strategy'],
                    fusion_phase=model.fusion.phase,epoch=epoch+1,optimizer_updates=(epoch+1)*231))
                rows,metrics,diagnostics = evaluate(model,test_loader,device)
                validate_metrics(metrics)
                export(root / 'epochs',f'epoch_{epoch+1:02d}',rows,metrics,diagnostics,epoch,root / 'last.pt')
                write_json(root / 'lora' / f'epoch_{epoch+1:02d}.json',lora_diagnostics(model))
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
                    optimizer_updates=(epoch+1)*231,fusion_base_sha256=training['fusion_base_sha256'],disabled_lora_sha256=training['disabled_lora_sha256'],test=dict(metrics=metrics)))
                write_json(root / 'history.json',history)
                best = select_best(root,model,config,epoch,metrics,best)
                write_json(root / 'process_memory.json',watcher.report())
                print(json.dumps(dict(epoch_complete=epoch+1,fusion_phase=model.fusion.phase,
                    test_macro_srocc=metrics['macro']['srocc'],best=best if math.isfinite(best) else None)),flush=True)
            if scheduler.last_epoch != 2310:
                raise AssertionError('Incomplete LR schedule')
            del optimizer,scheduler
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            write_json(root / 'status.json',dict(stage='evaluating_best_last',completed_epochs=10,optimizer_updates=2310))
            reload_checks = {}
            for kind in ('best','last'):
                checkpoint = root / f'{kind}.pt'
                payload = load_model_checkpoint(model,checkpoint,config)
                epoch = payload['epoch']
                del payload
                rows,metrics,diagnostics = evaluate(model,test_loader,device)
                original_report = json.loads((root / 'epochs' / f'epoch_{epoch+1:02d}_evaluation.json').read_text())
                original_rows = {(r['video_id'],r['task']):r['prediction'] for r in original_report['samples']}
                if len(original_rows) != 460:
                    raise ValueError('Incomplete reference epoch predictions')
                reload_error = max(abs(row['prediction']-original_rows[row['video_id'],row['task']]) for row in rows)
                if reload_error > 1e-6:
                    raise ValueError('Best/last full-test checkpoint reload error exceeds 1e-6')
                reload_checks[kind] = dict(epoch=epoch+1,predictions=len(rows),max_abs_error=reload_error)
                export(root,kind,rows,metrics,diagnostics,epoch,checkpoint)
                write_json(root / f'{kind}_lora.json',lora_diagnostics(model))
                write_json(root / f'{kind}_task_coefficients.json',task_coefficients(model))
                check_memory(watcher.report())
            write_json(root / 'checkpoint_reload_verification.json',dict(passed=True,checkpoints=reload_checks))
            write_json(root / 'process_memory.json',watcher.report())
            summarize(root)
            write_json(root / 'status.json',dict(stage='complete',variant=config['training_strategy'],
                completed_epochs=10,optimizer_updates=2310))
    except Exception as error:
        write_json(run / 'status.json',dict(stage='failed',variant=config['training_strategy'],error=repr(error),
            automatic_retry=False,last_checkpoint_saved=(root / 'last.pt').exists()))
        if watcher is not None:
            write_json(run / 'process_memory.json',watcher.report())
        raise
