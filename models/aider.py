"""AIDER: cross-branch FiLM, ordinary Transformer and private task LoRAs."""

import copy
import math

import torch
from torch import nn

from utils.initialization import (
    load_task_table,
    load_position_encoding,
    restore_encoder_initialization,
)
from engine.audit import digest
from utils.experiment import validate_model_config, film_video_condition
from .artifact_query import TechnicalExpert
from .backbones import VITALVisionEncoder, SlowFastEncoder, BLIPTextVideoEncoder
from .blip_temporal import BLIPTemporalSelfAttention
from .film import CrossBranchFiLM, video_mlp
from .fusion import TransformerFusion
from .registry import validate_task_ids
from .registry import TASKS, BRANCHES, ATTRIBUTES
from .task_private import task_linear, privatize_task_modules


class TaskEmbedding(nn.Module):
    """固定原始 CLIP 文本投影；不减均值，也不作 L2 单位化。"""

    def __init__(self, task):
        super().__init__()
        self.feature_dim = 768
        self.num_tasks = len(task['order'])
        for name, value in load_task_table(task).items():
            self.register_buffer(name, value)

    def forward(self, task_ids):
        validate_task_ids(task_ids, len(task_ids), self.num_tasks)
        return self.encoded_feature[task_ids]


class AIDER(nn.Module):
    def __init__(self, config, build_backbones=True):
        super().__init__()
        validate_model_config(config)
        model = config['model']
        self.film_video_condition = film_video_condition(config)
        self.frozen_private_heads = (model.get('head_policy', 'shared') == 'frozen_private'
                                     or 'head_rank' in config)
        self.private_heads = model.get('head_policy', 'shared') in ('private', 'frozen_private')
        video_condition_width = 256 if self.film_video_condition == 'video_only' else 1024
        linear_correction = self.film_video_condition == 'cross_branch_linear'
        condition_widths = (1024, 1280, 768) if linear_correction else (video_condition_width,) * 3
        self.config = copy.deepcopy(model)
        self.task = copy.deepcopy(config['task'])
        self.task_order = tuple(self.task['order'])
        self.experiment = copy.deepcopy(config['experiment'])
        self.variant = self.experiment['variant']
        self.ablation = copy.deepcopy(config['ablation'])
        self.training_strategy = config['training_strategy']
        self.active_tasks = tuple(self.ablation['tasks'])
        seed = config['training']['seed']
        if (
            self.ablation['depth'] != 2
            or (self.task_order != TASKS and not any(k in config for k in ('videofeedback','eduaigv')))
        ):
            raise ValueError('Expected the selected four-task depth-2 baseline')

        # 三个 Backbone：视觉权重冻结，BLIP 文本／交叉注意力继续训练。
        self.vital_backbone = None
        self.motion_backbone = None
        self.tv_backbone = None
        if build_backbones:
            self.vital_backbone = VITALVisionEncoder(**model['backbones']['vital'])
            self.motion_backbone = SlowFastEncoder(**model['backbones']['slowfast'])
            self.tv_backbone = BLIPTextVideoEncoder(**model['backbones']['blip'])

        # 保留旧属性名和注册顺序，保证检查点键名及优化器参数顺序不变。
        # *_projector 在当前模型中均为 FiLM，不是额外的维度投影。
        with torch.random.fork_rng(devices=[]):
            self.technical_expert = TechnicalExpert(1024, 512, **model['artifact_query_sigmoid'])
            self.technical_projector = CrossBranchFiLM(
                512, seed + 700000, seed + 820010, condition_widths[0], linear_correction
            )
            self.motion_projector = CrossBranchFiLM(
                256, seed + 700001, seed + 820011, condition_widths[1], linear_correction
            )
            self.motion_position_embedding = nn.Embedding(
                model['backbones']['slowfast']['max_positions'], 256
            )
            self.tv_projector = CrossBranchFiLM(
                768, seed + 700002, seed + 820012, condition_widths[2], linear_correction
            )
            self.dimension_encoder = TaskEmbedding(self.task)
            self.fusion = TransformerFusion(seed, load_position_encoding())
            self.blip_temporal = BLIPTemporalSelfAttention()

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 820020)
            self.alpha_gate = nn.Linear(768, 1)
            nn.init.zeros_(self.alpha_gate.weight)
            nn.init.constant_(self.alpha_gate.bias, math.log(0.1 / 0.9))

        # 每路独立的视频状态 MLP，只读取另外两路，不做自身特征置零占位。
        self.video_summaries = nn.ModuleDict(
            {} if linear_correction else {
                'technical': video_mlp(256 + 768, 512, 256, seed + 840000),
                'motion': video_mlp(512 + 768, 512, 256, seed + 840001),
                'text_video': video_mlp(512 + 256, 512, 256, seed + 840002),
            }
        )
        retained_count = restore_encoder_initialization(self, config)
        if self.private_heads:
            source_head = self.fusion.heads['shared']
            self.fusion.heads = nn.ModuleDict({task: copy.deepcopy(source_head) for task in self.task_order})
        self.private_task_modules = 'private_task_modules' in config
        if self.private_task_modules:
            privatize_task_modules(self)
        self.configure_trainable_parameters()
        self.task_embedding_hashes = {
            name: digest({name: value})
            for name, value in self.dimension_encoder.state_dict().items()
        }
        self.initialization_audit = {
            'seed': seed,
            'trained_checkpoint_loaded': False,
            'retained_snapshot_tensors': retained_count,
            'construction': 'depth2_dense_transformer_private_task_lora',
            'pool_count': 0,
            'heads': len(self.task_order) if self.private_heads else 1,
            'cross_branch_summary': ('raw_other_branch_temporal_means_no_summary_mlp'
                                     if linear_correction else 'three_independent_two_input_mlps_no_zero_padding'),
            'task_embedding_hashes': self.task_embedding_hashes,
            'film_video_condition': self.film_video_condition,
            'film_video_condition_width': (dict(zip(BRANCHES, condition_widths))
                                           if linear_correction else video_condition_width),
        }

    def configure_phase(self, phase):
        self.zero_grad(set_to_none=True)
        self.fusion.configure_phase(phase)

    def configure_trainable_parameters(self):
        if self.frozen_private_heads:
            self.requires_grad_(False)
            self.fusion.heads.requires_grad_(True)
            self.fusion.use_checkpoint = False
            if self.tv_backbone is not None:
                self.tv_backbone.model.text_encoder.encoder.gradient_checkpointing = False
            self.train(False)
            return
        self.requires_grad_(True)
        for backbone in (self.vital_backbone, self.motion_backbone):
            if backbone is not None:
                backbone.requires_grad_(False)
        if self.tv_backbone is not None:
            self.tv_backbone.configure_trainable_parameters()
        self.fusion.configure_phase('joint' if self.training_strategy == 'joint' else 'dense')

    def train(self, mode=True):
        if self.frozen_private_heads:
            super().train(False)
            self.fusion.heads.train(mode)
            return self
        super().train(mode)
        for backbone in (self.vital_backbone, self.motion_backbone):
            if backbone is not None:
                backbone.eval()
        if self.tv_backbone is not None:
            self.tv_backbone.train(mode)
        return self

    def forward(self, batch):
        task_ids = batch['task_id']
        validate_task_ids(task_ids, len(batch['prompt']), len(self.task_order))
        for row, task_id in enumerate(task_ids.detach().cpu().tolist()):
            task = self.task_order[task_id]
            if (
                batch['label_key'][row] != task
                or batch['dimension_text'][row] != self.task['descriptions'][task]
            ):
                raise ValueError('Task metadata mismatch')

        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_private_heads):
            raw, artifact = self.extract_raw(batch)
        result = self.forward_from_raw(raw, task_ids)
        result['artifact'] = artifact
        return result

    def extract_raw(self, batch):
        # Technical：patch 特征经 Artifact Prior 处理为 [B,8,512]。
        patches, grid = self.vital_backbone(batch['vital_frames'])
        technical, artifact = self.technical_expert(patches.float(), batch['artifact_prior'], grid)

        # Motion：[B,8,256]，添加原可训练时间位置编码。
        motion = self.motion_backbone(batch['slowfast_frames'])
        positions = torch.arange(motion.shape[1], device=motion.device)
        motion = motion + self.motion_position_embedding(positions)[None]

        # BLIP：逐帧跨模态 CLS → 原时序 SA → [B,8,768]。
        text_video = self.tv_backbone(batch['blip_frames'], list(batch['prompt']))
        text_video = self.blip_temporal(text_video)
        raw = {'technical': technical, 'motion': motion, 'text_video': text_video}
        return raw, artifact

    def video_context(self, raw, condition, task_ids=None):
        with torch.autocast(condition.device.type, enabled=False):
            technical = raw['technical'].float().mean(dim=1)
            motion = raw['motion'].float().mean(dim=1)
            text_video = raw['text_video'].float().mean(dim=1)
            if self.film_video_condition == 'cross_branch_linear':
                contexts = {
                    'technical': torch.cat([motion, text_video], dim=-1),
                    'motion': torch.cat([technical, text_video], dim=-1),
                    'text_video': torch.cat([technical, motion], dim=-1),
                }
                return None, contexts, task_linear(self.alpha_gate, condition.float(), task_ids).sigmoid()
            summaries = {
                'technical': self.video_summaries['technical'](
                    torch.cat([motion, text_video], dim=-1)
                ),
                'motion': self.video_summaries['motion'](
                    torch.cat([technical, text_video], dim=-1)
                ),
                'text_video': self.video_summaries['text_video'](
                    torch.cat([technical, motion], dim=-1)
                ),
            }
            contexts = {
                branch: summary if self.film_video_condition == 'video_only'
                else torch.cat([condition.float(), summary], dim=-1)
                for branch, summary in summaries.items()
            }
            alpha = task_linear(self.alpha_gate, condition.float(), task_ids).sigmoid()
        return summaries, contexts, alpha

    def forward_from_raw(self, raw, task_ids):
        validate_task_ids(task_ids, len(raw['technical']), len(self.task_order))
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_private_heads):
            tokens, routing = self.fuse_from_raw(raw, task_ids)
        frame_scores = self.score_frames(routing['fused_frames'], task_ids)
        routing['frame_scores'] = frame_scores
        return dict(predictions=frame_scores.mean(dim=1), frame_scores=frame_scores,
                    tokens=tokens, routing=routing)

    def score_frames(self, fused, task_ids):
        validate_task_ids(task_ids, len(fused), len(self.task_order))
        if not self.private_heads:
            return self.fusion.heads['shared'](fused).squeeze(-1)
        outputs, indices = [], []
        for index, task in enumerate(self.task_order):
            selected = torch.where(task_ids == index)[0]
            if selected.numel():
                outputs.append(self.fusion.heads[task](fused[selected]).squeeze(-1))
                indices.append(selected)
        return torch.cat(outputs)[torch.cat(indices).argsort()]

    def fuse_from_raw(self, raw, task_ids):
        condition = self.dimension_encoder(task_ids)
        summaries, contexts, alpha = self.video_context(raw, condition, task_ids)

        # 三路分别调制，再沿通道拼接：[B,8,512+256+768]。
        technical = self.technical_projector(
            raw['technical'], condition, contexts['technical'], alpha, task_ids
        )
        motion = self.motion_projector(raw['motion'], condition, contexts['motion'], alpha, task_ids)
        text_video = self.tv_projector(raw['text_video'], condition, contexts['text_video'], alpha, task_ids)
        tokens = {'technical': technical, 'motion': motion, 'text_video': text_video}
        features = torch.cat([technical, motion, text_video], dim=-1)

        # 两层共享 Transformer；评分 Head 的共享策略由配置决定。
        fused, routes = self.fusion(features, condition, task_ids)

        routing = {
            'pools': routes,
            'first_stage': {},
            'second_stage': {},
            'first_stage_applicable': False,
            'fused_frames': fused,
            'concatenated_features': features,
            'dimension_features': condition,
            'video_summary': summaries,
            'video_condition': contexts,
            'film_video_condition': self.film_video_condition,
            'alpha': alpha,
            'fusion_phase': self.fusion.phase,
            'adapters_enabled': self.fusion.phase != 'dense',
        }
        if self.film_video_condition == 'cross_branch_linear':
            routing['cross_branch_input'] = contexts
            routing['video_feature_change_rms'] = {}
            for branch, attr in zip(BRANCHES, ATTRIBUTES):
                with torch.no_grad():
                    modulation = getattr(self, attr).modulation(condition, contexts[branch], alpha, task_ids)
                    change = (raw[branch].float() * (alpha * modulation['delta_gamma'])[:, None]
                              + (alpha * modulation['delta_beta'])[:, None])
                    routing['video_feature_change_rms'][branch] = change.square().mean((1, 2)).sqrt()
        return tokens, routing

    def heads(self):
        yield from self.fusion.heads.items()

    def film_modules(self):
        return zip(BRANCHES, (getattr(self, name) for name in ATTRIBUTES))

    def compact_state_dict(self):
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        frozen = (
            'motion_backbone.feature_extraction.',
            'vital_backbone.vision_model.',
            'tv_backbone.model.',
        )
        state = {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith(frozen) or name in trainable
        }
        # 文本 buffer 在旧 compact checkpoint 中追加保存；保持相同顺序。
        state.update(
            {
                name: value
                for name, value in self.state_dict().items()
                if name.startswith('tv_backbone.model.text_encoder.')
            }
        )
        return state
