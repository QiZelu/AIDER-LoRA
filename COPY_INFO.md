# AIDER-LoRA 源码副本

对应实验：`aigvqa_db_transformer_task_lora_r8_raw_clip_two_stage5_5_seed42`。

- 已复制该实验实际使用的模型、backbone 封装、数据处理、训练、评估与工具源码，以及测试代码。
- 根目录 `config.yaml` 是该两阶段实验的原始配置；`README.md` 保留原实验说明。
- 源码内容保持原样，49 个运行源码文件已与实验 `sources.json` 的 SHA256 记录逐一核验。
- 未复制权重（含 `.pt`、`.pth`、`.ckpt`、`.safetensors`）、位置编码快照、Prior/特征/任务向量缓存、视频、训练日志和结果。
- 这是源码与配置归档，尚未做目录迁移适配。配置保留原工程中的数据、权重和缓存引用；普通训练入口还依赖原工程的 reference、artifacts 等资源，因此本目录不是独立运行包。两阶段实验原命令使用 `--variant two_stage`。
- `COPY_MANIFEST.json` 记录每个复制文件的来源和内容哈希。
