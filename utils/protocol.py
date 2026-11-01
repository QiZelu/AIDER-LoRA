"""Shared FiLM and task-conditioned shared LoRA; external resources are read-only."""
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from engine.audit import file_digest
from engine.io import write_json
from models.registry import TASKS
from utils.task_projection import TRANSFORM

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = Path(os.environ.get('AIDER_RESOURCE_ROOT',str(ROOT.parent / 'AIDER'))).resolve()
PARENT_RUNTIME = RESOURCE_ROOT / 'experiments/aigvqa_task_lora_fusion'
REFERENCE = PARENT_RUNTIME / 'reference'
ARTIFACTS = PARENT_RUNTIME / 'artifacts'
SOURCE = RESOURCE_ROOT / 'outputs/multitask/runs/aigvqa_db_transformer_task_lora_r8_raw_clip_two_stage5_5_seed42'
CONTROL = RESOURCE_ROOT / 'outputs/multitask/runs/aigvqa_db_shared_film_task_conditioned_lora_r8_raw_clip_residual_gate_joint_seed42'
CONTROL_CONFIG_SHA256 = 'b1c9e929b9be3ebd451245d9ede0b210562d3b75091c37efc8c0676e263c27f5'
CONTROL_SOURCES_SHA256 = '705e5c37d0ce96f323a560976506ad9e349895b004f57f398313c55e2ee9ec50'
OUTPUTS = {'two_stage':RESOURCE_ROOT / 'outputs/multitask/runs/aigvqa_db_shared_film_task_conditioned_lora_r8_raw_clip_residual_gate_two_stage5_5_seed42_aider_lora'}
CONTRACT = 'aider_aigvqa_shared_film_task_conditioned_shared_lora_raw_clip_residual_gate_two_stage5_5_r8_v1'
GPUS = {'two_stage':1}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reference_config():
    return yaml.safe_load((CONTROL / 'config.yaml').read_text())


def phase_for_epoch(config,epoch):
    if not 0 <= epoch < 10:
        raise ValueError('Expected epoch index 0..9')
    if config['training_strategy'] != 'two_stage':
        raise ValueError('This experiment requires dense epochs1-5 and adapter epochs6-10')
    return 'dense' if epoch < 5 else 'adapter'


def best_eligible(config,epoch):
    return phase_for_epoch(config,epoch) == 'adapter'


def make_config(root=None,gpu=None,variant='two_stage'):
    if variant not in OUTPUTS:
        raise ValueError('Expected the two_stage raw-CLIP residual-Gate variant')
    root = Path(root or OUTPUTS[variant]).resolve()
    gpu = GPUS[variant] if gpu is None else gpu
    cfg = reference_config()
    cfg['model']['readout_variant'] = 'shared_film_task_conditioned_lora'
    cfg['model']['task_film_policy'] = 'shared'
    cfg['model']['alpha_policy'] = 'shared'
    cfg['model']['task_lora'] = dict(rank=8,alpha=16,dropout=.05,bias=False,
        targets=['q_proj','k_proj','v_proj','out_proj','fc1','fc2'],
        sharing='shared_A_B_and_modulator_across_tasks_per_projection',
        condition='fixed_raw_clip_projection_768',selection='embedding_conditioning_no_task_parameter_bank',
        modulation=dict(kind='linear_residual',input_width=768,output_width=8,
                        precision='float32',initial_value=1.0,residual_initial_value=0.0,
                        weight_init='zeros',bias_init='zeros'),
        initialization='original_first_task_random_A_zero_B_isolated_rng')
    cfg.pop('private_task_modules',None)
    cfg['model']['task_embedding_transform'] = TRANSFORM
    cfg['sharing_policy'] = dict(task_film='shared_per_video_branch',video_film='shared_per_video_branch',
        video_summaries='shared_per_video_branch',film_alpha='shared_all_tasks',
        lora='shared_per_layer_and_projection',channel_modulator='shared_per_layer_and_projection',heads='task_private')
    # No cached tensors, weights or manifests are copied into the source tree.
    prep = SOURCE / 'preparation'
    cfg['task'].pop('intent_adapter',None)
    cfg['task'].update(table_path=str(prep / 'task_tables.json'),table_sha256=file_digest(prep / 'task_tables.json'),
        reference_table_path=str(REFERENCE / 'centered_task_tables.json'),
        reference_table_sha256=file_digest(REFERENCE / 'centered_task_tables.json'))
    cfg['data'].update(train_manifest=str(prep / 'train.jsonl'),val_manifest=str(prep / 'test.jsonl'))
    cfg['data']['artifact']['readonly_manifest'] = str(prep / 'prior_manifest.json')
    cfg['training']['output_dir'] = str(root)
    cfg['training_strategy'] = variant
    cfg['training_phases'] = [dict(phase='dense',epochs=[1,5]),dict(phase='adapter',epochs=[6,10])]
    cfg['phase_transition'] = dict(after_updates=1155,initialization='epoch5_last_not_test_best',
        preserve_other_adam_states=True,reset_learning_rate=False,warmup_again=False,
        freeze='fusion_shared_linears_biases_layernorms_and_final_norm_only',
        other_trainable_modules='unchanged',new_adapter_adam_state='empty',
        adapters='shared_A_B_and_channel_modulators')
    cfg['evaluation']['checkpoint'] = cfg['inference']['checkpoint'] = str(root / 'best.pt')
    cfg['evaluation']['best_epochs'] = [6,10]
    cfg['experiment'] = dict(contract=CONTRACT,variant=variant,physical_gpu=gpu,resource_root=str(RESOURCE_ROOT),
        code_root=str(ROOT),parent=str(CONTROL),parent_config_sha256=CONTROL_CONFIG_SHA256,
        task_embedding_transform=TRANSFORM,initialization='original_pretrained_and_untrained_seed42',
        checkpoint_includes_complete_blip_text=True,automatic_baseline_replacement=False,
        comparison_control=str(CONTROL))
    cfg['ablation'] = dict(contract=CONTRACT,variant=variant,gpu=gpu,tasks=list(TASKS),task_ids=list(range(4)),depth=2,epochs=10,warmup=116)
    cfg['baseline'] = dict(name='Shared FiLM and channel-residual shared LoRA with unmodified raw CLIP task embeddings',
        selection='Independent 5+5 ablation against raw CLIP joint; original split',
        runtime_architecture=dict(depth=2,width=1536,heads=12,ffn_hidden=1536,routed_experts=0,
                                  channel_modulators=12,heads_private=4,shared_lora_and_gate_parameters=368736))
    cfg['protocol'] = dict(contract=CONTRACT,variant=variant,output_root=str(root),resource_root=str(RESOURCE_ROOT),
        code_root=str(ROOT),task_table_sha256=cfg['task']['table_sha256'],task_embedding_transform=TRANSFORM,
        task_embedding_mean_subtraction=False,task_embedding_row_l2_normalization=False,intent_adapter_used=False,
        comparison_control=str(CONTROL),comparison_control_config_sha256=CONTROL_CONFIG_SHA256,
        comparison_control_sources_sha256=CONTROL_SOURCES_SHA256,
        train_sha256=file_digest(cfg['data']['train_manifest']),test_sha256=file_digest(cfg['data']['val_manifest']),
        prior_manifest_sha256=file_digest(prep / 'prior_manifest.json'),inventory_path=str(prep / 'video_inventory.json'),
        inventory_sha256=file_digest(prep / 'video_inventory.json'),projection_audit_path=str(prep / 'projection_audit.json'),
        projection_audit_sha256=file_digest(prep / 'projection_audit.json'),
        reference_snapshot_sha256=file_digest(PARENT_RUNTIME / 'reference_snapshot.json'),
        parent_sources_sha256=file_digest(SOURCE / 'sources.json'),
        parent_config_sha256=file_digest(SOURCE / 'config.yaml'),loss_protocol=cfg['loss_protocol'])
    return cfg


def validate_config(config):
    if config != make_config(config['protocol']['output_root'],config['experiment']['physical_gpu'],config['training_strategy']):
        raise ValueError('Shared FiLM/conditional LoRA configuration or run identity differs from pinned protocol')


def validate_reference(check_parent_code=False):
    provenance = json.loads((ROOT / 'SOURCE_PROVENANCE.json').read_text())
    allowed = {'utils/protocol.py','scripts/train_task_lora.py','scripts/audit_initialization.py',
               'engine/trainer.py','engine/preflight.py','config.yaml','SOURCE_PROVENANCE.json'}
    for name,expected in provenance['original_file_sha256'].items():
        if name not in allowed and file_digest(ROOT / name) != expected:
            raise ValueError('Unexpected change to the verified raw-CLIP joint implementation: '+name)
    if (ROOT / 'utils/intent_adapter.py').exists():
        raise ValueError('The raw CLIP experiment must not contain an intent projection module')
    if (file_digest(CONTROL / 'config.yaml') != CONTROL_CONFIG_SHA256
            or file_digest(CONTROL / 'sources.json') != CONTROL_SOURCES_SHA256):
        raise ValueError('Raw-CLIP joint comparison identity changed')
    status = json.loads((CONTROL/'status.json').read_text())
    history = json.loads((CONTROL/'history.json').read_text())
    if (status['stage'] != 'complete' or len(history) != 10 or history[-1]['optimizer_updates'] != 2310
            or any(row['fusion_phase'] != 'joint' for row in history)):
        raise ValueError('Raw-CLIP joint reference must have complete ten-epoch history')
    if check_parent_code:
        for name,item in json.loads((CONTROL/'sources.json').read_text()).items():
            if name.startswith(('runtime/','preparation/')) and file_digest(item['path']) != item['sha256']:
                raise ValueError('Raw-CLIP joint source/resource changed: '+name)
    snapshot = json.loads((PARENT_RUNTIME / 'reference_snapshot.json').read_text())
    for name,expected in snapshot.items():
        if file_digest(PARENT_RUNTIME / name) != expected:
            raise ValueError('Read-only reference changed: '+name)
    status = json.loads((SOURCE / 'status.json').read_text())
    history = json.loads((SOURCE / 'history.json').read_text())
    if status['stage'] != 'complete' or len(history) != 10 or history[-1]['optimizer_updates'] != 2310:
        raise ValueError('Private-LoRA two-stage comparison must be complete')
    recorded = json.loads((SOURCE / 'sources.json').read_text())
    for name,item in recorded.items():
        if name.startswith('preparation/') or (check_parent_code and name.startswith('runtime/')):
            if file_digest(item['path']) != item['sha256']:
                raise ValueError('Parent source/resource changed: '+name)
    if (SOURCE / 'config.yaml').read_bytes() != (SOURCE / 'preparation/config.yaml').read_bytes():
        raise ValueError('Parent configuration changed')


def validate_data(config):
    validate_config(config)
    prep = SOURCE / 'preparation'
    inventory_path = Path(config['protocol']['inventory_path'])
    if file_digest(inventory_path) != config['protocol']['inventory_sha256']:
        raise ValueError('Video inventory changed')
    inventory = json.loads(inventory_path.read_text())
    ids,prompts = {},{}
    for name,key,count in (('train','train_manifest',461),('test','val_manifest',115)):
        path = Path(config['data'][key])
        if path.read_bytes() != (REFERENCE / 'preparation' / (name+'.jsonl')).read_bytes() or file_digest(path) != config['protocol'][name+'_sha256']:
            raise ValueError('Original split, labels or order changed')
        rows = read_rows(path)
        if len(rows) != count or len({row['id'] for row in rows}) != count:
            raise ValueError('Wrong split coverage')
        for row in rows:
            if set(row['labels']) != set(TASKS) or not all(math.isfinite(float(v)) and 0 <= v <= 1 for v in row['labels'].values()):
                raise ValueError('Invalid original task labels')
            video = (Path(config['data']['video_root']) / row['video']).resolve()
            stat = video.stat()
            if not stat.st_size or inventory[row['id']] != dict(path=str(video),size=stat.st_size,mtime_ns=stat.st_mtime_ns):
                raise ValueError('Video changed: '+str(video))
        ids[name] = {row['id'] for row in rows}
        prompts[name] = {row['prompt'] for row in rows}
    if ids['train'] & ids['test'] or len(ids['train']|ids['test']) != 576:
        raise ValueError('Wrong original partition')
    prior = Path(config['data']['artifact']['readonly_manifest'])
    if prior.read_bytes() != (REFERENCE / 'preparation/prior_manifest.json').read_bytes() or file_digest(prior) != config['protocol']['prior_manifest_sha256']:
        raise ValueError('Readonly Prior manifest changed')
    cache = json.loads(prior.read_text())
    if not cache['readonly'] or len(cache['files']) != 576 or {v['video_id'] for v in cache['files'].values()} != ids['train']|ids['test']:
        raise ValueError('Wrong readonly Prior coverage')
    from utils.task_projection import load_task_table
    raw = load_task_table(config['task'])
    encoded = raw['encoded_feature']
    if (encoded.shape!=(4,768) or encoded.dtype!=torch.float32
            or not torch.equal(encoded,raw['original_feature'])
            or bool((encoded.norm(dim=-1)<=1.01).any()) or 'intent_adapter' in config['task']):
        raise ValueError('Task embeddings must be unmodified, non-unit raw CLIP projections')
    if (prep / 'task_tables.json').read_bytes() != (REFERENCE / 'preparation/task_tables.json').read_bytes():
        raise ValueError('Raw CLIP table changed')
    if file_digest(config['protocol']['projection_audit_path']) != config['protocol']['projection_audit_sha256']:
        raise ValueError('Projection audit changed')
    return dict(videos=dict(train=461,test=115),records=dict(train=1844,test=460),id_overlap=0,
        labels_unchanged=True,prior_files=576,task_embedding_transform=TRANSFORM,
        prompt_overlap=len(prompts['train'] & prompts['test']),resources_reused_readonly=True,
        intent_adapter_used=False,task_embedding_row_norms=encoded.norm(dim=1).tolist())


def audit_priors(config):
    from dataset.artifact_prior import ArtifactCache,ArtifactPriorExtractor
    options = dict(config['data']['artifact'])
    cache = ArtifactCache(options.pop('cache_dir'),readonly_manifest=options.pop('readonly_manifest'))
    extractor = ArtifactPriorExtractor(**options)
    def check(row):
        path = cache.path_for(Path(config['data']['video_root']) / row['video'],8,448,extractor)
        if cache.readonly.get(str(path.resolve()),{}).get('video_id') != row['id']:
            raise ValueError('Prior/video mismatch')
        value = cache.load(path)
        if value.shape != (8,448,448) or value.dtype != np.float32:
            raise ValueError('Invalid Prior geometry/dtype')
        return str(path)
    before = torch.get_rng_state().clone()
    rows = read_rows(config['data']['train_manifest'])+read_rows(config['data']['val_manifest'])
    with ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(check,rows))
    if len(set(paths)) != 576 or not torch.equal(before,torch.get_rng_state()):
        raise AssertionError('Prior audit coverage/RNG changed')
    return dict(videos=576,all_sha256_verified=True,readonly=True,cache_rebuilt=False,
        shape=[8,448,448],dtype='float32',model_rng_unchanged=True)


def prepare(root=None,gpu=None,variant='two_stage'):
    root = Path(root or OUTPUTS[variant]).resolve()
    gpu = GPUS[variant] if gpu is None else gpu
    if root.exists():
        cfg = load_prepared(root)
        if cfg['training_strategy'] != variant or cfg['experiment']['physical_gpu'] != gpu:
            raise ValueError('Prepared variant/GPU changed; use a fresh run')
        return cfg
    root.mkdir(parents=True,exist_ok=False)
    write_json(root / 'status.json',dict(stage='preparing',optimizer_updates=0))
    try:
        validate_reference(check_parent_code=True)
        before = torch.get_rng_state().clone()
        folder = root / 'preparation'
        folder.mkdir()
        cfg = make_config(root,gpu,variant)
        write_json(folder / 'data_audit.json',validate_data(cfg))
        write_json(folder / 'prior_audit.json',audit_priors(cfg))
        write_json(folder / 'resource_references.json',dict(readonly=True,
            task_table=cfg['task']['table_path'],train_manifest=cfg['data']['train_manifest'],
            test_manifest=cfg['data']['val_manifest'],prior_manifest=cfg['data']['artifact']['readonly_manifest'],
            initialization=cfg['training']['initialization'],positions=str(ARTIFACTS / 'depth2_video_adaptive_fixed_position.pt'),
            task_embedding_transform=TRANSFORM,intent_adapter_used=False))
        if not torch.equal(before,torch.get_rng_state()):
            raise AssertionError('Preparation consumed model RNG')
        text = yaml.safe_dump(cfg,sort_keys=False)
        (folder / 'config.yaml').write_text(text)
        (root / 'config.yaml').write_text(text)
        subprocess.run([sys.executable,'-B',str(ROOT / 'scripts/audit_initialization.py'),
            '--config',str(folder / 'config.yaml'),'--output-dir',str(folder / 'initialization_comparison')],
            check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        write_json(folder / 'complete.json',dict(contract=CONTRACT,variant=variant,
            config_sha256=file_digest(folder / 'config.yaml'),model_rng_unchanged=True))
        write_json(root / 'status.json',dict(stage='prepared',formal_training_started=False,optimizer_updates=0))
        return cfg
    except Exception as error:
        write_json(root / 'status.json',dict(stage='preparation_failed',error=repr(error),automatic_retry=False))
        raise


def load_prepared(root=None):
    root = Path(root or OUTPUTS['two_stage']).resolve()
    path = root / 'preparation/config.yaml'
    meta = json.loads((root / 'preparation/complete.json').read_text())
    if meta['contract'] != CONTRACT or file_digest(path) != meta['config_sha256'] or (root / 'config.yaml').read_bytes() != path.read_bytes():
        raise ValueError('Prepared configuration changed')
    config = yaml.safe_load(path.read_text())
    if config['protocol']['output_root'] != str(root) or meta['variant'] != config['training_strategy']:
        raise ValueError('Wrong run identity')
    paired = json.loads((root / 'preparation/initialization_comparison/paired_initialization_audit.json').read_text())
    if not paired.get('passed') or not paired.get('model_rng_identical') or not paired.get('all_other_state_tensors_identical'):
        raise ValueError('Two-stage/joint fresh initialization comparison did not pass')
    validate_reference()
    validate_data(config)
    return config


def selected_source_hashes(config):
    validate_data(config)
    validate_reference(check_parent_code=True)
    result = {}
    recorded = json.loads((SOURCE / 'sources.json').read_text())
    for name,item in recorded.items():
        if name.startswith('pretrained/'):
            if file_digest(item['path']) != item['sha256']:
                raise ValueError('Pretrained source changed: '+name)
            result[name] = item
    paths = [SOURCE / 'config.yaml',SOURCE / 'sources.json',PARENT_RUNTIME / 'reference_snapshot.json']
    paths += [CONTROL / name for name in ('config.yaml','sources.json','best_evaluation.json','last_evaluation.json')]
    paths += list(REFERENCE.rglob('*'))+list(ARTIFACTS.rglob('*'))
    for path in sorted(paths):
        if path.is_file():
            result['resource/'+str(path.relative_to(RESOURCE_ROOT))] = dict(path=str(path),sha256=file_digest(path))
    for folder in ('models','dataset','engine','utils','scripts'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            result['runtime/'+str(path.relative_to(ROOT))] = dict(path=str(path),sha256=file_digest(path))
    for name in ('baseline_initialization.json','config.yaml','COPY_MANIFEST.json','SOURCE_PROVENANCE.json'):
        path = ROOT / name
        result['runtime/'+name] = dict(path=str(path),sha256=file_digest(path))
    for path in sorted((Path(config['protocol']['output_root']) / 'preparation').rglob('*')):
        if path.is_file():
            result['preparation/'+path.name] = dict(path=str(path),sha256=file_digest(path))
    return result
