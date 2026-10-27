"""可选全局梯度裁剪；无裁剪时只检查和统计，不改写 grad。"""

import math

import torch


def gradient_clip_limit(config):
    value = config.get('training', {}).get('gradient_clip', 1.0)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('gradient_clip must be null or a finite positive number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError('gradient_clip must be null or a finite positive number')
    return float(value)


def prepare_gradients(model, config, context=''):
    limit = gradient_clip_limit(config)
    if limit is not None:
        # 保持原参数遍历及归约方式，避免改变已有实验的更新轨迹。
        torch.nn.utils.clip_grad_norm_(model.parameters(), limit, error_if_nonfinite=True)
        return None

    with torch.no_grad():
        squared_norms = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f'Nonfinite gradient at {context}: {name}')
            squared_norms.append(grad.double().square().sum())
        norm = float(torch.stack(squared_norms).sum().sqrt()) if squared_norms else 0.0
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite global gradient norm at ' + context)
    return dict(global_norm_fp64=norm, gradient_clip=None,
                applied_scale=1.0, exceeds_1=norm > 1.0,
                active_parameter_tensors=len(squared_norms))


def gradient_statistics(records):
    values = torch.tensor([r['global_norm_fp64'] for r in records], dtype=torch.float64)
    if not values.numel():
        raise ValueError('No gradient records')
    quantiles = torch.quantile(values, torch.tensor([.5, .9, .99], dtype=torch.float64))
    return dict(updates=len(records), median=float(quantiles[0]),
                p90=float(quantiles[1]), p99=float(quantiles[2]),
                maximum=float(values.max()), exceeds_1_fraction=float((values > 1).double().mean()),
                applied_scale=1.0, gradient_clip=None)


def validate_checkpoint_gradient_policy(payload, config):
    saved_limit = gradient_clip_limit(payload['config'])
    if payload.get('gradient_clip', saved_limit) != saved_limit:
        raise ValueError('Checkpoint gradient policy metadata mismatch')
    if saved_limit != gradient_clip_limit(config):
        raise ValueError('Checkpoint gradient clipping policy mismatch')
