AIDER-Modulation — current Full baseline architecture and training

Contents: model source, vendored backbone model definitions, minimal model-only
configuration helpers, fixed raw CLIP task vectors and fixed position buffers.
The training/ package and train.py provide standalone data loading, loss, AdamW,
5+5 scheduling, preflight, training and evaluation. No run outputs, learned
checkpoints, backbone weights, caches or dataset manifests are included.

Architecture:
VITAL patches -> spatial projection + Prior-guided Query residual -> [B,8,512]
SlowFast -> learned temporal positions -> [B,8,256]
BLIP framewise prompt/video CLS -> temporal SA -> [B,8,768]
Three shared FiLM maps, then channel concat -> [B,8,1536]
Two Pre-LN Transformer blocks,12 heads,128 channels/head,FFN1536.
Each layer has a separate task-shared Linear(768,128): Q'=Q*(1+g(z)).
Four private Heads:1536->256->64->16->1; frame scores averaged per video.
FiLM: x'=(1+gamma_task(z)+alpha(z)*delta_gamma(video))*x
          +beta_task(z)+alpha(z)*delta_beta(video).
Each branch's video context uses the other two branches. Raw CLIP task vectors
are not centered or L2-normalized. Fixed task order: static,temporal,dynamic,text_video.

Usage (from this directory, in the existing uve-copy Python environment):
  from aider_modulation import load_config, build_aider
  config = load_config(pretrained_root='/path/to/pretrained_models')
  model = build_aider(config)
  model.configure_phase('adapter')  # QK gates active, shared Fusion base frozen
  model.eval()
  # Caller supplies preprocessed batch tensors and aligned task metadata.
  output = model(batch)

batch contains vital_frames[B,8,3,448,448],slowfast_frames[B,3,8,224,224],
blip_frames[B,8,3,224,224],artifact_prior[B,8,448,448],prompt[list[str]],
task_id[B] int64,label_key[list[str]],dimension_text[list[str]].
Inputs must follow the original normalized video preprocessing/Prior definition.
Dataset/preprocessing support is included; its inputs and caches remain external.
Use CUDA BF16 autocast for real backbone inference, matching the baseline.

For core-only feature fusion without loading backbone weights:
  model = build_aider(load_config(),build_backbones=False).eval()
  model.configure_phase('adapter')
  output = model.forward_from_raw(raw,task_ids)
raw keys:technical[B,8,512],motion[B,8,256],text_video[B,8,768].
Task/head IDs remain aligned; the model initializes with untrained parameters.

The original configure_phase('dense'/'adapter'/'joint') methods and parameter names
are preserved. The training entry implements the baseline 5+5 optimizer/scheduler.
The default initial phase is dense. For exact historical untrained projector/SA
initialization, pass an external initialization_path to load_config(); no such file
is shipped. No trained model is provided by this source-only export.

External dependencies: PyTorch,NumPy,Transformers,timm,einops,safetensors,fairscale,
PyTorchVideo (for deserializing the external SlowFast checkpoint), OpenCV, PyYAML,
tqdm. Use the existing uve-copy environment; dependency versions are inherited
from the source experiments, rather than upgraded by this export.
External backbone directories/files:VITAL-Vision-Encoder-300M,slowfast.pth,
model_large.pth,bert-large-uncased. The fixed task table avoids loading CLIP at runtime.
The vendor backbone source headers retain original attribution/license notices.

Source: AIDER/experiments/aigvqa_qk_diag_residual_two_stage5_5/models.
This is the same architecture as Full in the five-split ablation suite.
Core forward equations/state names are unchanged; only experiment-specific imports
are replaced by local model-only helpers. SOURCE_MANIFEST.json records file hashes.


TRAINING
========
Run commands from this directory using the existing uve-copy Python environment.
All relative file paths, including paths inside YAML, resolve against this directory.
The default configuration references existing external data, Prior caches, backbone
weights and the original UNTRAINED seed42 initialization snapshot. It does not
load any completed experiment checkpoint. Adjust configs/train.yaml if relocating.
The trainer imports all architecture/training code locally; AIDER paths in YAML
supply data/resources only. Copying this directory alone does not copy those inputs.

  python -B train.py --help
  python -B train.py --mode check
  python -B train.py --mode preflight --gpu 0 --output-dir outputs/preflight_seed42
  python -B train.py --mode train --gpu 0 --output-dir outputs/full_seed42
  python -B train.py --mode evaluate --gpu 0 --run-dir outputs/full_seed42 --checkpoint outputs/full_seed42/best.pt
  python -B train.py --mode summarize --run-dir outputs/full_seed42

check is the default and is CPU-only: it verifies manifests, external weights,
initialization hash, task constants and read-only Prior cache content. It does not
create a run or initialize CUDA. preflight performs disposable real CUDA updates
(two batch8s in each phase) and stops. train always runs its own disposable preflight
and then constructs a fresh model with matched parameters/CPU/CUDA RNG; use a new
output directory for each invocation. Existing runs are never overwritten or
resumed automatically. Only an explicit --mode train starts formal training.
No scheduler/background job is automatically started by this export.

Default Full-baseline settings:
- AIGVQA-DB static, temporal, dynamic, text_video, four private scoring Heads.
- The current generator-stratified seed42 split: 456 train / 120 test videos,
  1824 / 480 task records, 228 mixed batch8 updates per epoch. Original manifests
  are referenced externally without reshuffling. Alternate manifests must have
  the same four normalized labels and disjoint train/test videos.
- Eight frames, batch8, no accumulation, no gradient clipping, 10 epochs.
- Model seed42; independent CPU sampling seed9042; no dropped tail samples.
- Epochs1-5: fusion base trainable, QK gates disabled/frozen.
- Epochs6-10: fusion base frozen, residual QK gates enabled/trainable.
  FiLM, video adapters, BLIP text/SA and private Heads continue training.
- AdamW main LR1e-4, BLIP text LR1e-5, weight decay0.05, FP32 parameters/Adam
  states and Smooth L1(beta0.1), BF16 forward. No MI/PLCC/rank auxiliary loss.
- Cosine across all10 epochs; 5% warmup (114 updates for default split).
  The phase transition preserves continuing Adam states and current LR.
- Visual backbones frozen. Full fine-tuned BLIP text and frozen Fusion base
  are included in each newly generated compact checkpoint.
- 14GiB PyTorch allocator limit and 16GiB per-process driver memory budget.
  Missing/changed cached Priors fail without regeneration. Failures do not
  trigger retry, batch reduction or stopping any other process.

Console output uses progress bars. When explicitly trained, the selected output
folder receives configuration/input hashes, preflight and initialization audits,
per-epoch metrics, sample orders, gradient/loss/FiLM/QK/attention diagnostics,
stage1_last.pt, last.pt, best.pt, and best/last long and wide predictions.
Best is the earliest strict improvement in four-task macro test SROCC during
EPOCHS6-10 ONLY, matching the source 5+5 baseline. The same held-out set is used
for selection and reporting, so this is not an independent final test estimate.
Last is always epoch10. Reloaded predictions must agree within1e-6.

Checkpoints are bound to this standalone contract, exact configuration, split,
task table and source/resource hashes. --checkpoint accepts checkpoints produced
by this trainer for the specified run; historical experiment checkpoints use a
different contract and cannot be loaded via this training CLI. Core API behavior
and state names remain unchanged. Training resume is deliberately unsupported.
