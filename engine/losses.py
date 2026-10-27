"""FP32 Smooth L1 only: ordinary Transformer fusion has no router/MI objective."""
import math
import torch
from torch.nn import functional as F


def regression_loss(predictions, targets, beta=0.1):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('Invalid Smooth L1 beta')
    if predictions.ndim != 1 or targets.shape != predictions.shape or not predictions.numel():
        raise ValueError('Expected nonempty aligned scores')
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise FloatingPointError('Nonfinite scores or targets')
    with torch.autocast(predictions.device.type,enabled=False):
        values = F.smooth_l1_loss(predictions.float(),targets.float(),beta=beta,reduction='none')
    return values.mean(), values


def quality_loss_protocol(config):
    expected = dict(regression=dict(kind='smooth_l1',beta=0.1,reduction='sample_mean'),
                    auxiliary=None,combination='regression_only',precision='float32')
    if config['loss_protocol'] != expected or config['training'].get('moe_mi_weight',0) != 0:
        raise ValueError('Dense/LoRA training requires regression only')
    return expected


def quality_training_loss(model,result,batch,config):
    quality_loss_protocol(config)
    loss, values = regression_loss(result['predictions'],batch['target'],0.1)
    return loss, values, dict(regression_loss=float(loss.detach()),total_loss=float(loss.detach()))


class TrainingLossSummary:
    def __init__(self):
        self.samples = self.updates = 0
        self.sample_sum = self.update_sum = 0.0

    def add(self,samples,diagnostics):
        self.samples += samples.numel()
        self.updates += 1
        self.sample_sum += float(samples.detach().sum())
        self.update_sum += diagnostics['total_loss']

    def report(self):
        if not self.updates:
            raise ValueError('No training updates')
        return dict(regression_loss_sample_mean=self.sample_sum/self.samples,
                    total_loss_update_mean=self.update_sum/self.updates,
                    samples=self.samples,optimizer_updates=self.updates,phase='before_optimizer_step')
