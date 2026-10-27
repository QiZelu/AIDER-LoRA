"""Active FiLM Transformer baseline and its fixed task registry."""

TASKS = ("static", "temporal", "dynamic", "text_video")
BRANCHES = ("technical", "motion", "text_video")
ATTRIBUTES = ("technical_projector", "motion_projector", "tv_projector")
WIDTHS = (512, 256, 768)
# Historical serialization identifier, paired with a single supported variant.
BASELINE_NAME = "multimodal_readout_variants"
BASELINE_VARIANT = "transformer_task_lora"


def validate_task_ids(ids, batch_size, num_tasks=len(TASKS)):
    import torch
    if (ids is None or ids.dtype != torch.int64 or ids.shape != (batch_size,)
            or not batch_size or not bool(((ids >= 0) & (ids < num_tasks)).all())):
        raise ValueError('Expected valid int64 task IDs [B]')
