"""恒等初始化的任务基础调制与跨分支视频残差调制。"""

import torch
from torch import nn
from .task_private import task_linear


def video_mlp(input_width, hidden_width, output_width, seed, zero_output=False):
    """视频状态／调制 MLP；隔离初始化 RNG，避免改变其他模块初值。"""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        module = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
        )
        for layer in (module[0], module[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if zero_output:
            nn.init.zeros_(module[2].weight)
    return module


def video_linear(input_width, output_width, seed):
    """原维度跨分支输入直接产生修正，零初始化且不消耗全局 RNG。"""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        module = nn.Linear(input_width, output_width)
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)
    return module


class CrossBranchFiLM(nn.Module):
    """任务调制 + 其他两路视频特征提供的残差调制。"""

    def __init__(self, width, task_seed, adaptive_seed, video_condition_width=1024,
                 linear_correction=False):
        super().__init__()
        if not linear_correction and video_condition_width not in (256, 1024):
            raise ValueError("FiLM expects a 256D video or 1024D task-video condition")
        self.width = width
        self.video_condition_width = video_condition_width
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(task_seed)
            self.affine = nn.Linear(768, 2 * width)
            nn.init.zeros_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)
        self.adaptive = (
            video_linear(video_condition_width, 2 * width, adaptive_seed)
            if linear_correction else video_mlp(
                video_condition_width, 256, 2 * width, adaptive_seed, zero_output=True
            )
        )

    def coefficients(self, condition, task_ids=None):
        with torch.autocast(condition.device.type, enabled=False):
            return task_linear(self.affine, condition.float(), task_ids).chunk(2, dim=-1)

    def modulation(self, condition, video_condition, alpha, task_ids=None):
        if video_condition.shape != (len(condition), self.video_condition_width):
            raise ValueError("Unexpected FiLM video condition shape")
        with torch.autocast(condition.device.type, enabled=False):
            gamma, beta = self.coefficients(condition, task_ids)
            delta_gamma, delta_beta = self.adaptive(video_condition.float()).chunk(
                2, dim=-1
            )
            return {
                "task_gamma": gamma,
                "task_beta": beta,
                "delta_gamma": delta_gamma,
                "delta_beta": delta_beta,
                "effective_gamma": 1 + gamma + alpha * delta_gamma,
                "effective_beta": beta + alpha * delta_beta,
            }

    def forward(self, features, condition, video_condition, alpha, task_ids=None):
        modulation = self.modulation(condition, video_condition, alpha, task_ids)
        gamma = modulation["effective_gamma"][:, None]
        beta = modulation["effective_beta"][:, None]
        # [B,T,D]：同一视频的所有帧使用同一组调制系数。
        return features.float() * gamma + beta
