"""Pinned AIGVQA four-task data and independent dense/LoRA run identities."""
import copy
import json
import math
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
WORKSPACE = ROOT.parents[1]
REFERENCE = ROOT / 'reference'
SOURCE = WORKSPACE / 'outputs/multitask/runs/film_video_only_private_task_modules_shared_gate_no_clip_mi03_raw_clip_projection_seed42'
CONTRACT = 'aider_aigvqa_dense_transformer_task_lora_raw_clip_r8_v1'
OUTPUTS = {variant: WORKSPACE / 'outputs/multitask/runs' / name for variant, name in {
    'joint': 'aigvqa_db_transformer_task_lora_r8_raw_clip_joint_seed42',
    'two_stage': 'aigvqa_db_transformer_task_lora_r8_raw_clip_two_stage5_5_seed42',
}.items()}
GPUS = {'joint': 0, 'two_stage': 3}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reference_config():
    return yaml.safe_load((REFERENCE / 'config.yaml').read_text())


def phase_for_epoch(config, epoch):
    if not 0 <= epoch < 10:
        raise ValueError('Expected epoch index 0..9')
    return 'joint' if config['training_strategy'] == 'joint' else ('dense' if epoch < 5 else 'adapter')


def best_eligible(config, epoch):
    return config['training_strategy'] == 'joint' or epoch >= 5


def make_config(root=None, gpu=None, variant='joint'):
    if variant not in OUTPUTS:
        raise ValueError('Expected joint or two_stage')
    root = Path(root or OUTPUTS[variant]).resolve()
    gpu = GPUS[variant] if gpu is None else gpu
    cfg = reference_config()
    cfg['task'].update(table_path=str(root / 'preparation/task_tables.json'),
        table_sha256=file_digest(root / 'preparation/task_tables.json'),
        reference_table_path=str(REFERENCE / 'centered_task_tables.json'),
        reference_table_sha256=file_digest(REFERENCE / 'centered_task_tables.json'))
    cfg['model'].pop('two_stage_moe', None)
    cfg['model'].pop('moe_gate_policy', None)
    cfg['model']['readout_variant'] = 'transformer_task_lora'
    cfg['model']['readout_spec'] = dict(width=1536, depth=2, heads=12, ffn_hidden=1536,
        dropout=0.0, activation='gelu', block='u=x+Attention(LN1(x)); y=u+FFN(LN2(u))',
        final_norm=True, fixed_sinusoidal_positions=True, checkpoint_use_reentrant=False)
    cfg['model']['task_lora'] = dict(rank=8, alpha=16, dropout=0.05, bias=False,
        targets=['q_proj','k_proj','v_proj','out_proj','fc1','fc2'],
        selection='explicit_task_id', initialization='independent_random_A_zero_B')
    cfg['data'].update(train_manifest=str(root / 'preparation/train.jsonl'),
        val_manifest=str(root / 'preparation/test.jsonl'))
    cfg['data']['artifact']['readonly_manifest'] = str(root / 'preparation/prior_manifest.json')
    cfg['training'].pop('moe_mi_weight', None)
    cfg['training']['output_dir'] = str(root)
    cfg['training_strategy'] = variant
    cfg['training_phases'] = ([dict(phase='joint', epochs=[1,10])] if variant == 'joint' else
        [dict(phase='dense', epochs=[1,5]), dict(phase='adapter', epochs=[6,10])])
    cfg['phase_transition'] = dict(after_updates=1155, initialization='epoch5_last_not_test_best',
        preserve_other_adam_states=True, reset_learning_rate=False, warmup_again=False,
        freeze='fusion_shared_linears_biases_layernorms_and_final_norm_only',
        other_trainable_modules='unchanged', new_adapter_adam_state='empty') if variant == 'two_stage' else None
    cfg['loss_protocol'] = dict(regression=dict(kind='smooth_l1',beta=0.1,reduction='sample_mean'),
        auxiliary=None, combination='regression_only', precision='float32')
    cfg['evaluation']['checkpoint'] = cfg['inference']['checkpoint'] = str(root / 'best.pt')
    cfg['evaluation']['best_epochs'] = [1,10] if variant == 'joint' else [6,10]
    cfg['experiment'] = dict(contract=CONTRACT, variant=variant, physical_gpu=gpu,
        parent=str(SOURCE), parent_config_sha256=file_digest(REFERENCE / 'config.yaml'),
        task_embedding_transform=TRANSFORM, initialization='original_pretrained_and_untrained_seed42',
        checkpoint_includes_complete_blip_text=True, automatic_baseline_replacement=False)
    cfg['ablation'] = dict(contract=CONTRACT,variant=variant,gpu=gpu,tasks=list(TASKS),
        task_ids=list(range(4)),depth=2,epochs=10,warmup=116)
    cfg['private_task_modules'] = dict(task_modulators=12,alpha_generators=4,heads=4,
        task_lora_sets=4,task_selection='explicit_task_id')
    cfg['baseline'] = dict(name='Ordinary shared Transformer with independent task LoRAs',
        selection='Independent ablation; original split; not a Baseline replacement',
        runtime_architecture=dict(depth=2,width=1536,heads=12,ffn_hidden=1536,routed_experts=0,gates=0))
    cfg['protocol'] = dict(contract=CONTRACT,variant=variant,output_root=str(root),
        task_table_sha256=cfg['task']['table_sha256'],task_embedding_transform=TRANSFORM,
        train_sha256=file_digest(cfg['data']['train_manifest']),test_sha256=file_digest(cfg['data']['val_manifest']),
        prior_manifest_sha256=file_digest(root / 'preparation/prior_manifest.json'),
        inventory_sha256=file_digest(root / 'preparation/video_inventory.json'),
        projection_audit_sha256=file_digest(root / 'preparation/projection_audit.json'),
        reference_snapshot_sha256=file_digest(ROOT / 'reference_snapshot.json'),
        source_snapshot_sha256=file_digest(ROOT / 'source_snapshot.json'), loss_protocol=cfg['loss_protocol'])
    return cfg


def validate_config(config):
    if config != make_config(config['protocol']['output_root'], config['experiment']['physical_gpu'],
                             config['training_strategy']):
        raise ValueError('Dense/LoRA architecture, loss, training or run identity differs from pinned configuration')


def validate_reference(check_parent_code=False):
    snapshot = json.loads((ROOT / 'reference_snapshot.json').read_text())
    for name, expected in snapshot.items():
        if file_digest(ROOT / name) != expected:
            raise ValueError('Pinned reference changed: ' + name)
    if check_parent_code:
        source = json.loads((ROOT / 'source_snapshot.json').read_text())
        for path, expected in source['parent_code_sha256'].items():
            if file_digest(path) != expected:
                raise ValueError('Parent runtime changed: ' + path)


def validate_data(config):
    validate_config(config)
    folder = Path(config['protocol']['output_root']) / 'preparation'
    inventory = json.loads((folder / 'video_inventory.json').read_text())
    if file_digest(folder / 'video_inventory.json') != config['protocol']['inventory_sha256']:
        raise ValueError('Video inventory changed')
    ids, prompts = {}, {}
    for name, key, count in (('train','train_manifest',461),('test','val_manifest',115)):
        path = Path(config['data'][key])
        if (path.read_bytes() != (REFERENCE / 'preparation' / (name + '.jsonl')).read_bytes()
                or file_digest(path) != config['protocol'][name + '_sha256']):
            raise ValueError('Original split, labels or order changed')
        rows = read_rows(path)
        if len(rows) != count or len({row['id'] for row in rows}) != count:
            raise ValueError('Wrong split coverage')
        for row in rows:
            if set(row['labels']) != set(TASKS) or not all(math.isfinite(float(v)) and 0 <= v <= 1 for v in row['labels'].values()):
                raise ValueError('Original task labels invalid')
            video = (Path(config['data']['video_root']) / row['video']).resolve()
            stat = video.stat()
            if not stat.st_size or inventory[row['id']] != dict(path=str(video),size=stat.st_size,mtime_ns=stat.st_mtime_ns):
                raise ValueError('Video changed: ' + str(video))
        ids[name] = {row['id'] for row in rows}
        prompts[name] = {row['prompt'] for row in rows}
    if ids['train'] & ids['test'] or len(ids['train'] | ids['test']) != 576:
        raise ValueError('Wrong original partition')
    prior = folder / 'prior_manifest.json'
    if (prior.read_bytes() != (REFERENCE / 'preparation/prior_manifest.json').read_bytes()
            or file_digest(prior) != config['protocol']['prior_manifest_sha256']):
        raise ValueError('Readonly Prior manifest changed')
    cache = json.loads(prior.read_text())
    if not cache['readonly'] or len(cache['files']) != 576 or {v['video_id'] for v in cache['files'].values()} != ids['train'] | ids['test']:
        raise ValueError('Wrong Prior coverage')
    from utils.task_projection import load_task_table
    load_task_table(config['task'])
    if ((folder / 'task_tables.json').read_bytes() != (REFERENCE / 'preparation/task_tables.json').read_bytes()
            or file_digest(folder / 'projection_audit.json') != config['protocol']['projection_audit_sha256']):
        raise ValueError('Raw CLIP table or extraction audit changed')
    return dict(videos=dict(train=461,test=115),records=dict(train=1844,test=460),id_overlap=0,
        labels_unchanged=True,prior_files=576,task_embedding_transform=TRANSFORM,
        prompt_overlap=len(prompts['train'] & prompts['test']))


def audit_priors(config):
    from dataset.artifact_prior import ArtifactCache, ArtifactPriorExtractor
    options = dict(config['data']['artifact'])
    cache = ArtifactCache(options.pop('cache_dir'),readonly_manifest=options.pop('readonly_manifest'))
    extractor = ArtifactPriorExtractor(**options)
    def check(row):
        path = cache.path_for(Path(config['data']['video_root']) / row['video'],8,448,extractor)
        if cache.readonly.get(str(path.resolve()),{}).get('video_id') != row['id']:
            raise ValueError('Prior/video mismatch')
        value = cache.load(path)
        if value.shape != (8,448,448) or value.dtype != np.float32:
            raise ValueError('Invalid Prior geometry or dtype')
        return str(path)
    before = torch.get_rng_state().clone()
    rows = read_rows(config['data']['train_manifest']) + read_rows(config['data']['val_manifest'])
    with ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(check,rows))
    if len(set(paths)) != 576 or not torch.equal(before,torch.get_rng_state()):
        raise AssertionError('Prior audit coverage/RNG changed')
    return dict(videos=576,all_sha256_verified=True,readonly=True,cache_rebuilt=False,
        shape=[8,448,448],dtype='float32',model_rng_unchanged=True)


def prepare(root=None, gpu=None, variant='joint'):
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
        for name in ('train.jsonl','test.jsonl','prior_manifest.json','task_tables.json','projection_audit.json','video_inventory.json'):
            (folder / name).write_bytes((REFERENCE / 'preparation' / name).read_bytes())
        cfg = make_config(root,gpu,variant)
        write_json(folder / 'data_audit.json',validate_data(cfg))
        write_json(folder / 'prior_audit.json',audit_priors(cfg))
        if not torch.equal(before,torch.get_rng_state()):
            raise AssertionError('Preparation consumed model RNG')
        text = yaml.safe_dump(cfg,sort_keys=False)
        (folder / 'config.yaml').write_text(text)
        (root / 'config.yaml').write_text(text)
        write_json(folder / 'complete.json',dict(contract=CONTRACT,variant=variant,
            config_sha256=file_digest(folder / 'config.yaml'),model_rng_unchanged=True))
        write_json(root / 'status.json',dict(stage='prepared',formal_training_started=False,optimizer_updates=0))
        return cfg
    except Exception as error:
        write_json(root / 'status.json',dict(stage='preparation_failed',error=repr(error),automatic_retry=False))
        raise


def load_prepared(root=None):
    root = Path(root or OUTPUTS['joint']).resolve()
    path = root / 'preparation/config.yaml'
    meta = json.loads((root / 'preparation/complete.json').read_text())
    if (meta['contract'] != CONTRACT or file_digest(path) != meta['config_sha256']
            or (root / 'config.yaml').read_bytes() != path.read_bytes()):
        raise ValueError('Prepared configuration changed')
    config = yaml.safe_load(path.read_text())
    if config['protocol']['output_root'] != str(root) or meta['variant'] != config['training_strategy']:
        raise ValueError('Wrong run identity')
    validate_reference()
    validate_data(config)
    return config


def selected_source_hashes(config):
    validate_data(config)
    validate_reference(check_parent_code=True)
    result = {}
    for name, item in json.loads((REFERENCE / 'sources.json').read_text()).items():
        if name.startswith('pretrained/'):
            if file_digest(item['path']) != item['sha256']:
                raise ValueError('Pretrained source changed: ' + name)
            result[name] = item
    paths = list((ROOT / 'reference').rglob('*')) + list((ROOT / 'artifacts').rglob('*'))
    for folder in ('models','dataset','engine','utils','scripts'):
        paths += list((ROOT / folder).rglob('*.py'))
    paths += [ROOT / 'source_snapshot.json', ROOT / 'reference_snapshot.json']
    paths += list((Path(config['protocol']['output_root']) / 'preparation').rglob('*'))
    for path in sorted(paths):
        if path.is_file():
            key = ('runtime/' + str(path.relative_to(ROOT))) if path.is_relative_to(ROOT) else ('preparation/' + path.name)
            result[key] = dict(path=str(path),sha256=file_digest(path))
    return result
