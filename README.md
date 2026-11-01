# AIDER-LoRA：raw CLIP＋残差 Gate＋5+5

此目录对应最新的 `aigvqa_db_shared_film_task_conditioned_lora_r8_raw_clip_residual_gate_two_stage5_5_seed42` 实现，包含模型源码、训练／评估入口、配置和测试。

## 模型

- 任务表示：原始 CLIP 文本投影（768维），不经过 Intent Adapter、去中心化或 L2 归一化；任务表固定。
- 三分支 FiLM、视频修正和 α 网络保留原共享方式；视觉 Backbone 冻结，BLIP 文本和时序 SA 继续训练。
- Fusion 为两层共享普通 Pre-LN Transformer，宽度1536、12个注意力头，每层的 Q/K/V/Out/FC1/FC2 加共享 LoRA，共12处。
- LoRA 为 `W0*x + (alpha/r)*B*diag(1+Linear(768,8)(z))*A*x`，rank8、alpha16、dropout0.05。A 随机初始化，B 和 Gate 的权重／偏置为0；初始通道权重为1。
- 四个任务各有一个私有 `1536→256→64→16→1` Head，前三层后使用 ReLU，逐帧评分后对8帧取均值。

## 5+5 训练

第1～5轮关闭并冻结 LoRA A/B 与 Gate。第6～10轮启用 LoRA，仅冻结 Fusion 基础线性层、偏置、LayerNorm 和最终归一化层；FiLM、α、视频摘要、私有 Head、BLIP 文本、时序 SA 等原可训练模块持续更新。

第1155次更新后直接延续第5轮 last，保留持续训练参数的 Adam 状态，新 LoRA 状态为空；不重启学习率或再次预热。总计10轮、2310步，batch8，尾批4。损失仅为 FP32 Smooth L1（β=0.1）。每轮测试，best 仅从第6～10轮按四任务宏平均 SROCC 选择，同分保留较早轮次。

## 资源与使用

默认资源根目录为同级 `AIDER`。预训练权重、未训练 seed42 初始化、任务表、Prior、数据及历史对照通过该目录只读引用。本源码包不携带 `.pt`、`.pth` 或其他缓存文件。依赖清单为 `requirements.txt`，BLIP 配置副本位于 `configs/med_config.json`；实际配置保留来源实验已核验的 AIDER 资源路径。

在本目录运行：

```bash
python -B scripts/train_task_lora.py --help
python -B scripts/train_task_lora.py --mode prepare
python -u -B scripts/train_task_lora.py --mode train --gpu 1 --background
python -B -m unittest discover -s tests -v
```

使用 `--resource-root /path/to/AIDER` 或 `AIDER_RESOURCE_ROOT` 指定其他资源位置，命令行参数优先；使用 `--output-dir` 指定新的实验目录。默认输出目录为来源实验名称后附加 `_aider_lora` 的独立目录。训练入口拒绝覆盖已有训练，也不会自动重试或停止其他 GPU 进程；每进程上限16 GiB。此次源码替换本身不启动训练。

可用模式为 `prepare|preflight|train|evaluate|summarize`，variant 为 `two_stage`。预检包含两个阶段共12个真实 batch8，正式训练前重新初始化。完成后自动评估 best／last、导出460行长表及115行宽表，并与 raw CLIP joint 对照比较。

Checkpoint 绑定配置、源码、实验目录和训练阶段；已有实验继续使用原封存入口及配置评估，本目录创建独立的新实验。

## 来源与回滚

`SOURCE_EXPERIMENT.json` 记录本包来源、默认配置和旧目录备份；`PACKAGE_MANIFEST.json` 记录本包文件哈希；`package_verification.json` 记录复制后的验证结果。`SOURCE_PROVENANCE.json` 和 `COPY_MANIFEST.json` 保留来源实现的历史溯源数据，后者不是本次包的文件清单。`reference/` 保存来源配置、源码清单和原实现验证报告的只读副本。
