"""Extract and validate fixed raw CLIP text projections without centering or L2."""
import json
from pathlib import Path

import torch

from engine.audit import digest, file_digest
from engine.io import write_json

TRANSFORM = 'fp32_raw_clip_text_projection_identity'


def extract_projection(model_dir, tasks, descriptions):
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        tokenizer = CLIPTokenizer.from_pretrained(model_dir, local_files_only=True)
        encoder = CLIPTextModelWithProjection.from_pretrained(model_dir, local_files_only=True).float().eval()
        inputs = tokenizer([descriptions[t] for t in tasks], padding=True, truncation=True,
            max_length=tokenizer.model_max_length, return_tensors='pt')
        # This is the output of CLIP's text_projection, before any row scaling.
        raw = encoder(**inputs).text_embeds.float().detach().cpu()
    if not torch.equal(before, torch.get_rng_state()): raise AssertionError('CLIP extraction consumed model RNG')
    if raw.shape != (4,768) or not torch.isfinite(raw).all(): raise ValueError('Invalid raw CLIP projection')
    return raw, {key:value.tolist() for key,value in inputs.items()}


def create_task_table(folder, base, reference, source_hashes):
    folder = Path(folder)
    tasks, descriptions = base['task']['order'], base['task']['descriptions']
    # Verify the exact CLIP artifacts used by the centered MI control.
    clip = {k:v for k,v in source_hashes.items() if k.startswith('pretrained/clip/')}
    if not clip: raise ValueError('Missing pinned CLIP source hashes')
    for key,item in clip.items():
        if file_digest(item['path']) != item['sha256']: raise ValueError('CLIP source changed: '+key)
    raw, tokens = extract_projection(base['model']['dimension_encoder']['model_dir'],tasks,descriptions)
    old = json.loads(Path(reference).read_text())
    unit = torch.nn.functional.normalize(raw,dim=1)
    unit_delta = float((unit-torch.tensor(old['original'],dtype=torch.float32)).abs().max())
    centered = torch.nn.functional.normalize(unit-unit.mean(0,keepdim=True),dim=1)
    centered_delta = float((centered-torch.tensor(old['transformed'],dtype=torch.float32)).abs().max())
    if max(unit_delta,centered_delta) > 1e-6: raise ValueError('CLIP extraction differs from the original task encoding')
    values = dict(encoded_feature=raw, original_feature=raw, task_mean=torch.zeros(1,768),
                  centered_norms=raw.norm(dim=1,keepdim=True))
    table = dict(task_order=tasks, descriptions=descriptions, transform=TRANSFORM,
        mean_subtraction=False, row_l2_normalization=False,
        raw_projection=raw.tolist(), transformed=raw.tolist(), original=raw.tolist(),
        mean=values['task_mean'].tolist(), centered_norms=values['centered_norms'].tolist(),
        legacy_buffer_semantics=dict(task_mean='zero: no shift applied',centered_norms='raw row norms for diagnostics only; never divide'),
        hashes={key:digest({key:value}) for key,value in values.items()},
        reference_table_sha256=file_digest(reference), clip_sources=clip, tokenization=tokens)
    write_json(folder/'task_tables.json',table)
    write_json(folder/'projection_audit.json',dict(transform=TRANSFORM,shape=list(raw.shape),dtype=str(raw.dtype),
        raw_projection_sha256=digest({'raw_projection':raw}),raw_row_norms=dict(zip(tasks,raw.norm(dim=1).tolist())),
        normalized_raw_vs_reference_original_max_abs_delta=unit_delta,
        reconstructed_centered_vs_reference_max_abs_delta=centered_delta,
        normalization_only_for_reference_validation=True,model_input_is_raw_projection=True,
        model_rng_unchanged=True,clip_sources=clip,reference_table_sha256=file_digest(reference),
        cosine_matrix=(unit@unit.T).tolist()))


def load_task_table(task):
    path = Path(task['table_path'])
    if file_digest(path) != task['table_sha256']: raise ValueError('Task table content changed')
    saved = json.loads(path.read_text())
    if (saved['task_order'] != task['order'] or saved['descriptions'] != task['descriptions']
            or saved['transform'] != TRANSFORM or saved['mean_subtraction'] is not False
            or saved['row_l2_normalization'] is not False): raise ValueError('Wrong raw CLIP task transform/registry')
    values = {key:torch.tensor(saved[source],dtype=torch.float32) for key,source in
        [('encoded_feature','transformed'),('original_feature','original'),('task_mean','mean'),('centered_norms','centered_norms')]}
    raw = torch.tensor(saved['raw_projection'],dtype=torch.float32)
    for name,value in values.items():
        if not torch.isfinite(value).all() or digest({name:value}) != saved['hashes'][name]:
            raise ValueError('Invalid raw task tensor: '+name)
    if raw.shape != (4,768) or not torch.equal(raw,values['encoded_feature']) or not torch.equal(raw,values['original_feature']):
        raise ValueError('Model task features must equal the unmodified raw projection')
    if values['task_mean'].shape != (1,768) or torch.count_nonzero(values['task_mean']):
        raise ValueError('Raw projection must apply no task mean shift')
    if not torch.equal(values['centered_norms'],raw.norm(dim=1,keepdim=True)):
        raise ValueError('Raw norm diagnostic changed')
    if bool((raw.norm(dim=1) <= 1.01).any()): raise ValueError('Unexpected unit-normalized task table')
    reference = Path(task['reference_table_path'])
    if file_digest(reference) != task['reference_table_sha256'] or saved['reference_table_sha256'] != task['reference_table_sha256']:
        raise ValueError('Centered reference task table changed')
    old = json.loads(reference.read_text())
    if not torch.allclose(torch.nn.functional.normalize(raw,dim=1),torch.tensor(old['original']),atol=1e-6,rtol=0):
        raise ValueError('Raw task vector directions differ from the original CLIP encoding')
    return values
