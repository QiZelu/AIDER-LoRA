# AIGVQA-DB: shared ordinary Transformer and independent task LoRAs

Two independent experiments use the latest raw-CLIP task table without task-mean subtraction or L2 normalization. The four tasks, private FiLM/alpha/Heads, video-only cross-branch FiLM, backbone freeze policy, full BLIP text fine-tuning and temporal SA match the pinned parent. No trained MoE checkpoint initializes these experiments.

The fusion input stays `[B,8,1536]`. Each of two Pre-LN Transformer layers has 12-head self-attention, its residual, then `LayerNorm -> Linear(1536,1536) -> GELU -> Linear(1536,1536)` and a second residual. Sinusoidal positions and final LayerNorm are retained. There are no expert banks, Gates or additional always-on FFNs. Explicit Q/K/V and output projections ensure every LoRA actually executes, including checkpoint recomputation.

Every task owns independent A/B matrices on each layer's Q, K, V, attention output, fc1 and fc2: `W_t = W_shared + (16/8) B_t A_t`. Rank8, alpha16 and dropout0.05 follow MoE-Rater's actual stage-two shell configuration and its Attention+FFN injection scope. A is independently seeded; B is zero. At initialization all adapter deltas are zero by design. Each task adds 294,912 parameters (1,179,648 for four tasks); biases and LayerNorm have no LoRA. Mixed batches dispatch only present tasks and restore original row order, so absent task parameters have no gradients or Adam/weight-decay updates. Selection uses explicit task IDs, not CLIP similarity or a router.

- `joint`, physical CUDA0: shared Transformer plus task LoRAs train for all ten epochs.
- `two_stage`, physical CUDA3: epochs1-5 train the ordinary Transformer with all LoRAs bypassed/frozen. After update1155, freeze only fusion base linears, biases, block LayerNorms and final LayerNorm, activate the original zero-delta task LoRAs, and continue epochs6-10. FiLM, alpha, private Heads, BLIP text, temporal SA and every other originally trainable module continue training. Their Adam moments and global LR step are preserved; LoRA Adam states begin empty. There is no scheduler restart or repeated warmup. The switch uses epoch5 last, never test-best. `stage1_last.pt` is an immutable hard link to the complete atomic epoch-five checkpoint, and later last.pt writes replace the path rather than modify this inode.

Both variants start from identical untrained seed42 parameters and CPU/CUDA RNG. Retained non-fusion state matches the untrained raw-CLIP parent (121 tensors); pretrained files are verified by SHA256. Independent local CPU sampling reproduces all ten original permutations (seeds9042-9051). Original 461/115 videos, 1844/460 task records, eight-frame anchor policy and 576 read-only Prior caches are unchanged; missing/changed caches fail rather than rebuild. Batch8 has a four-record tail, 231 updates/epoch, 2310 total. AdamW LR1e-4, BLIP text1e-5, weight decay0.05, 116-step warmup then one cosine schedule. BF16 forward, FP32 trainable parameters/Adam moments, no accumulation or clipping, four data-loader workers.

The objective is FP32 Smooth L1 beta0.1 only. Task-expert MI is undefined without experts and is removed, with no replacement auxiliary loss. Reports separate sample-mean regression from update-mean loss.

Use `/home/u202511081000084/miniconda3/envs/uve-copy/bin/python` from this directory:

```bash
python -B scripts/train_task_lora.py --mode prepare --variant joint
python -B scripts/train_task_lora.py --mode prepare --variant two_stage
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B -m unittest discover -s tests -v
python -u -B scripts/train_task_lora.py --mode train --variant joint --gpu 0 --background
python -u -B scripts/train_task_lora.py --mode train --variant two_stage --gpu 3 --background
```

Modes: `prepare|preflight|train|evaluate|summarize`; arguments: `--variant joint|two_stage`, `--gpu` physical index, `--output-dir`, evaluate-only `--checkpoint`, train-only `--background`. Defaults are independent `outputs/multitask/runs/aigvqa_db_transformer_task_lora_r8_raw_clip_joint_seed42` and `aigvqa_db_transformer_task_lora_r8_raw_clip_two_stage5_5_seed42`. Existing run/launch artifacts are never overwritten. Startup requires an idle selected GPU and at least16GiB free; PyTorch allocation is capped at14GiB and driver process memory must remain below16GiB. No eviction, reassignment, reduced batch or automatic retry. A failure affects only its own worker.

Real preflight checks six disposable batch8 updates per training phase (two mixed, four single-task); two-stage also verifies the optimizer/scheduler transition. It checks gradients, frozen parameters/buffers, absent task adapter/FiLM/Head parameter and Adam invariance, FP32 states, driver memory and full BLIP-text/base/adapter checkpoint restoration within1e-6. Initial A gradients can be zero while B is zero. Preflight updates are discarded and seed42 rebuilt before formal training. CPU tests cover plain-Transformer equivalence, task isolation/reordering, checkpointing recomputation and inactive Adam behavior, deterministic paired initialization, all sample permutations, phase-transition state preservation and checkpoint identity rejection.

Every epoch tests all four tasks. Best uses strict improvement of test macro SROCC, ties keep the earlier epoch. Joint candidates are epochs1-10; two-stage candidates are adapter epochs6-10 only. Epochs1-5 remain in history and exports. After ten epochs, best/last are reloaded and evaluated, exporting 460-row long/115-row wide predictions, PLCC/SROCC/KROCC/RMSE, FiLM diagnostics, per-projection task LoRA delta norms/cosines, and gradient diagnostics. Strict checkpoints include the entire fusion base even when frozen, all task LoRAs and complete fine-tuned BLIP text, and bind the variant, task registry/table, split, source hashes, phase and objective. Old MoE or other-variant/run checkpoints are rejected.

`summary.json` is created only after that run's complete checks and best/last exports. Once both summaries exist, `verification/comparison_metrics.csv` and `comparison.json` compare them. Test data participate in model selection, the best candidate windows differ, the model seed is fixed, and removal of MoE also removes MI. These experiments do not automatically replace the Baseline. Existing runtimes, results and other training processes are untouched.
