"""Prior-guided artifact query used by the AIDER technical expert."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PriorGuidedArtifactQuery(nn.Module):
    """Aggregate projected VITAL patches with learned queries and log-prior bias."""

    def __init__(
        self,
        d_model: int,
        num_queries: int = 2,
        lambda_init: float = 0.1,
        lambda_max: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not math.isfinite(lambda_init):
            raise ValueError("lambda_init must be finite")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if not math.isfinite(lambda_max) or lambda_max <= 0.0:
            raise ValueError("lambda_max must be finite and positive")
        self.d_model = int(d_model)
        self.num_queries = int(num_queries)
        self.lambda_max = float(lambda_max)
        self.eps = float(eps)

        self.queries = nn.Parameter(torch.empty(num_queries, d_model))
        if not 0.0 < lambda_init < self.lambda_max:
            raise ValueError("lambda_init must be strictly between zero and lambda_max")
        probability = float(lambda_init) / self.lambda_max
        theta_init = math.log(probability / (1.0 - probability))
        self.theta = nn.Parameter(
            torch.full((num_queries,), theta_init, dtype=torch.float32)
        )
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.zeros_(self.pool[-1].weight)
        nn.init.zeros_(self.pool[-1].bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        artifact_prior: torch.Tensor,
        patch_grid: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, timesteps, num_patches, width = patch_tokens.shape
        if width != self.d_model:
            raise ValueError(f"Expected D={self.d_model}, got {width}")
        if num_patches != patch_grid[0] * patch_grid[1]:
            raise ValueError(f"N={num_patches} does not match patch grid {patch_grid}")
        if artifact_prior.ndim != 4:
            raise ValueError(
                "artifact_prior must be [B,T,H,W], got "
                f"{artifact_prior.shape}"
            )
        if artifact_prior.shape[:2] != (batch, timesteps):
            raise ValueError("Artifact prior is not aligned with VITAL timesteps")

        content_logits = torch.einsum(
            "kd,btnd->btkn", self.queries, patch_tokens
        ) / math.sqrt(self.d_model)
        patch_prior = F.adaptive_avg_pool2d(
            artifact_prior.reshape(
                batch * timesteps, 1, *artifact_prior.shape[-2:]
            ),
            patch_grid,
        ).reshape(batch, timesteps, num_patches)
        log_prior = torch.log(patch_prior.clamp_min(self.eps))
        prior_strength = self.lambda_max * torch.sigmoid(self.theta)
        prior_logits = (
            prior_strength.view(1, 1, self.num_queries, 1)
            * log_prior.unsqueeze(2)
        )
        attention = torch.softmax(content_logits + prior_logits, dim=-1)
        query_tokens = torch.einsum("btkn,btnd->btkd", attention, patch_tokens)
        query_weights = torch.softmax(self.pool(query_tokens).squeeze(-1), dim=-1)
        artifact_feature = torch.einsum(
            "btk,btkd->btd", query_weights, query_tokens
        )
        return artifact_feature, {
            "prior_strength": prior_strength,
            "patch_prior": patch_prior,
            "artifact_attention": attention,
            "query_weights": query_weights,
        }


class TechnicalExpert(nn.Module):
    """Build one 512D technical-quality token per sampled frame."""

    def __init__(
        self,
        vital_dim: int,
        d_model: int,
        num_queries: int,
        lambda_init: float = 0.1,
        lambda_max: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.patch_projector = nn.Sequential(
            nn.Linear(vital_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.spatial_projector = nn.Sequential(
            nn.Linear(vital_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.artifact_query = PriorGuidedArtifactQuery(
            d_model=d_model,
            num_queries=num_queries,
            lambda_init=lambda_init,
            lambda_max=lambda_max,
            eps=eps,
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.residual_gate[-1].weight)
        nn.init.constant_(self.residual_gate[-1].bias, -2.0)

    def forward(
        self,
        raw_patches: torch.Tensor,
        artifact_prior: torch.Tensor,
        patch_grid: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, _, num_patches, _ = raw_patches.shape
        if num_patches != patch_grid[0] * patch_grid[1]:
            raise ValueError("Raw VITAL patches do not match patch_grid")
        spatial_token = self.spatial_projector(raw_patches.mean(dim=2))
        projected_patches = self.patch_projector(raw_patches)
        artifact_feature, diagnostics = self.artifact_query(
            projected_patches, artifact_prior, patch_grid
        )
        residual_gate = torch.sigmoid(
            self.residual_gate(torch.cat([spatial_token, artifact_feature], dim=-1))
        )
        technical = self.output_norm(
            spatial_token + residual_gate * artifact_feature
        )
        diagnostics.update(
            {
                "artifact_feature": artifact_feature,
                "residual_gate": residual_gate,
                "spatial_token": spatial_token,
            }
        )
        return technical, diagnostics
