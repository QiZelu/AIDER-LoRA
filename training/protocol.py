"""Standalone Full-baseline training contract and external input validation."""
import copy
import hashlib
import json
import math
from pathlib import Path
import yaml
from aider_modulation import TASKS, build_aider, load_config
from aider_modulation.support import validate_model_config
from training.engine.audit import digest, file_digest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'aider_modulation_full_two_stage_v1'
BASE_HASH = '9d81688ea9feaf67c93499f53c3714d84dc396d8eaec4d28ddb61c6e7649c9db'
RNG_HASH = 'deebeee84a1da43276837bc0f376ae8b90228c67f7889a8b1549291375d984e1'


def resolve(path):
    p = Path(path).expanduser()
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def phase_for_epoch(config, epoch):
    if not 0 <= epoch < 10:
        raise ValueError('Expected zero-based epoch 0..9')
    return 'dense' if epoch < 5 else 'adapter'


def best_eligible(config, epoch):
    return 5 <= epoch < 10


def config_digest(config):
    value = copy.deepcopy(config)
    value['protocol'].pop('config_sha256', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = [ROOT / 'train.py']
    for folder in ('aider_modulation', 'training'):
        paths += sorted((ROOT / folder).rglob('*.py'))
    paths += sorted((ROOT / 'configs').glob('*.json'))
    return {str(p.relative_to(ROOT)): file_digest(p) for p in paths}


def derived_counts(train_n, test_n):
    steps = math.ceil(train_n * 4 / 8)
    return dict(train_videos=train_n, test_videos=test_n, train_records=train_n*4,
                test_records=test_n*4, steps_per_epoch=steps, total_updates=steps*10,
                transition_updates=steps*5)


def make_config(path, output_dir=None, gpu=0):
    options = yaml.safe_load(resolve(path).read_text())
    if gpu < 0:
        raise ValueError('GPU must be a nonnegative physical index')
    cfg = load_config(resolve(options['pretrained_root']), resolve(options['initialization']))
    cfg['training'].update(options['training'])
    cfg['training']['initialization'] = str(resolve(options['initialization']))
    cfg['training']['output_dir'] = str(resolve(output_dir or options['output_dir']))
    cfg['data'] = copy.deepcopy(options['data'])
    for key in ('train_manifest', 'val_manifest', 'video_root'):
        cfg['data'][key] = str(resolve(cfg['data'][key]))
    for key in ('cache_dir', 'readonly_manifest'):
        cfg['data']['artifact'][key] = str(resolve(cfg['data']['artifact'][key]))
    cfg['loader'] = options['loader']
    cfg['counts'] = derived_counts(len(read_rows(cfg['data']['train_manifest'])),
                                  len(read_rows(cfg['data']['val_manifest'])))
    cfg['training']['warmup_steps'] = math.ceil(cfg['counts']['total_updates'] * cfg['training']['warmup_ratio'])
    cfg['ablation']['task_ids'] = list(range(4))
    cfg['experiment'].update(contract=CONTRACT, physical_gpu=gpu)
    cfg['evaluation'] = dict(best_epochs=[6,10])
    cfg['loss_protocol'] = dict(regression=dict(kind='smooth_l1', beta=.1, reduction='sample_mean'),
                               auxiliary=None, combination='regression_only', precision='float32')
    resources = {}
    backbone = cfg['model']['backbones']
    files = [Path(cfg['training']['initialization']),
             Path(cfg['data']['artifact']['readonly_manifest']),
             Path(backbone['slowfast']['checkpoint_path']), Path(backbone['blip']['checkpoint_path'])]
    for folder in (backbone['vital']['model_dir'], backbone['blip']['bert_path']):
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files += sorted(p for p in folder.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(folder).parts))
    for p in files:
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(p)
        resources[str(p)] = file_digest(p)
    if resources[cfg['training']['initialization']] != options['initialization_sha256']:
        raise ValueError('External untrained initialization snapshot changed')
    cfg['protocol'] = dict(contract=CONTRACT, output_root=cfg['training']['output_dir'],
        train_sha256=file_digest(cfg['data']['train_manifest']),
        test_sha256=file_digest(cfg['data']['val_manifest']), resources=resources,
        sources=source_hashes())
    cfg['protocol']['config_sha256'] = config_digest(cfg)
    validate_config(cfg)
    return cfg


def validate_config(config):
    validate_model_config(config)
    if config['protocol']['contract'] != CONTRACT or config_digest(config) != config['protocol']['config_sha256']:
        raise ValueError('Configuration identity changed')
    if config['protocol']['output_root'] != config['training']['output_dir']:
        raise ValueError('Output root mismatch')
    t = config['training']; d = config['data']; loader = config['loader']
    required = dict(epochs=10, gradient_accumulation=1, loss_kind='smooth_l1',
                    smooth_l1_beta=.1, precision='bfloat16', gradient_clip=None)
    if any(t.get(k) != v for k,v in required.items()) or t.get('warmup_ratio') != .05:
        raise ValueError('Full baseline requires 10 epochs, FP32 Smooth L1, BF16 forward, no accumulation/clipping and 5% warmup')
    if loader['video_batch_size'] != 8 or loader['drop_last'] or not d['use_prior']:
        raise ValueError('Expected batch8, complete sampling and Prior enabled')
    if (d['num_frames'],d['min_total_frames'],d['vital_size'],d['expert_size']) != (8,8,448,224):
        raise ValueError('Baseline video preprocessing changed')
    c = config['counts']
    if c != derived_counts(c['train_videos'],c['test_videos']) or t['warmup_steps'] != math.ceil(c['total_updates']*.05):
        raise ValueError('Update counts or warmup mismatch')
    from training.engine.losses import quality_loss_protocol
    quality_loss_protocol(config)


def validate_data(config):
    validate_config(config)
    data = config['data']; ids = []; paths = []; prompts = []; quotas = {}
    from collections import Counter
    for split,key,n,hashkey in [('train','train_manifest',config['counts']['train_videos'],'train_sha256'),
                                ('test','val_manifest',config['counts']['test_videos'],'test_sha256')]:
        if file_digest(data[key]) != config['protocol'][hashkey]:
            raise ValueError('Dataset manifest changed: '+key)
        rows = read_rows(data[key]); current = set(); videos = set()
        if len(rows) != n or n < 8:
            raise ValueError('Each split needs at least eight videos')
        for row in rows:
            if row['id'] in current or set(row['labels']) != set(TASKS):
                raise ValueError('Duplicate video ID or wrong labels')
            if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in row['labels'].values()):
                raise ValueError('Expected four finite normalized MOS values in [0,1]')
            if not isinstance(row['prompt'], str):
                raise ValueError('Expected video prompt string')
            p = (Path(data['video_root']) / row['video']).resolve()
            if not p.is_file() or not p.stat().st_size:
                raise FileNotFoundError(p)
            if str(p) in videos:
                raise ValueError('Duplicate video path')
            current.add(row['id']); videos.add(str(p))
        ids.append(current); paths.append(videos); prompts.append({r['prompt'] for r in rows})
        quotas[split] = dict(Counter(Path(r['video']).parent.name for r in rows))
    if ids[0] & ids[1] or paths[0] & paths[1]:
        raise ValueError('Training/test video overlap')
    from aider_modulation.support import load_task_table
    load_task_table(config['task'])
    return dict(train_videos=len(ids[0]), test_videos=len(ids[1]), video_overlap=0,
                prompt_overlap=len(prompts[0] & prompts[1]), generator_counts=quotas,
                manifest_hashes_verified=True)


def audit_priors(config):
    import torch
    from training.dataset.artifact_prior import ArtifactCache, ArtifactPriorExtractor
    options = dict(config['data']['artifact'])
    cache = ArtifactCache(options.pop('cache_dir'), readonly_manifest=options.pop('readonly_manifest'))
    extractor = ArtifactPriorExtractor(**options)
    before = torch.get_rng_state().clone(); count = 0
    for key in ('train_manifest','val_manifest'):
        for row in read_rows(config['data'][key]):
            path = cache.path_for(Path(config['data']['video_root'])/row['video'],8,448,extractor)
            value = cache.load(path)
            if value is None:
                raise FileNotFoundError(path)
            if cache.readonly.get(str(path.resolve()),{}).get('video_id') != row['id'] or value.shape != (8,448,448) or str(value.dtype) != 'float32':
                raise ValueError('Invalid Prior video binding/geometry: '+str(path))
            count += 1
    if not torch.equal(before,torch.get_rng_state()):
        raise AssertionError('Prior validation consumed model RNG')
    return dict(videos=count, all_sha256_verified=True, readonly=True, cache_rebuilt=False)


def selected_source_hashes(config):
    validate_config(config)
    if source_hashes() != config['protocol']['sources']:
        raise ValueError('Training or model source changed since preparation')
    for path, expected in config['protocol']['resources'].items():
        if file_digest(path) != expected:
            raise ValueError('External resource changed: '+path)
    return dict(runtime=config['protocol']['sources'], resources=config['protocol']['resources'])


def prepare_run(config):
    from training.engine.io import write_json
    data = validate_data(config); prior = audit_priors(config)
    root = Path(config['training']['output_dir'])
    root.mkdir(parents=True,exist_ok=False)
    (root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(root/'data_audit.json',data); write_json(root/'prior_audit.json',prior)
    write_json(root/'status.json',dict(stage='prepared',optimizer_updates=0))
    return root


def load_prepared(root):
    cfg = yaml.safe_load((Path(root)/'config.yaml').read_text())
    if Path(root).resolve() != Path(cfg['protocol']['output_root']).resolve():
        raise ValueError('Saved run must be loaded from its original directory')
    validate_data(cfg)
    return cfg


def initialize_model(config, build_backbones=True):
    import torch
    from aider_modulation.qk_gate import gate_items
    validate_config(config)
    model = build_aider(config,build_backbones=build_backbones).train()
    model.case = 'full'; model.branches = ('technical','motion','text_video'); model.fusion_width = 1536
    rng = digest({'cpu_rng':torch.get_rng_state()})
    # Match the parent Transformer names before LoRA wrappers were removed.
    def canonical(name):
        parts = name.split('.')
        if len(parts) > 2 and parts[-2] in ('q_proj','k_proj','v_proj','out_proj','fc1','fc2'):
            parts.insert(-1,'base')
        return '.'.join(parts)
    base_equal = digest({canonical(n):p for n,p in model.fusion.base_parameters()}) == BASE_HASH
    if not base_equal or rng != RNG_HASH:
        raise AssertionError('Full baseline untrained initialization or construction RNG changed')
    gates = list(gate_items(model))
    if len(gates) != 2 or any(torch.count_nonzero(p) for _,g in gates for p in g.parameters()):
        raise AssertionError('Expected two zero-initialized QK gates')
    model.construction_rng_sha256 = rng
    model.initialization_audit.update(initial_state_sha256=digest(model.compact_state_dict()),
        construction_rng_sha256=rng, shared_transformer_base_equal_parent=base_equal,
        channel_modulator_parameters=sum(p.numel() for _,g in gates for p in g.parameters()),
        trained_checkpoint_loaded=False, lora_parameters=0, private_heads=4)
    return model
