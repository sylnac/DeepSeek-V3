# Copyright (c) 2025 DeepSeek-V3 Fork Authors
# Licensed under the MIT License (see LICENSE-CODE)
"""
DeepSeek-V3 Model Architecture

This module implements the core transformer architecture for DeepSeek-V3,
including Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE) layers.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """Configuration arguments for DeepSeek-V3 model."""
    vocab_size: int = 102400
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    score_func: str = "softmax"
    route_scale: float = 1.0
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    dtype: str = "bf16"
    max_seq_len: int = 4096


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotary position embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MoEGate(nn.Module):
    """Mixture-of-Experts gating mechanism with top-k routing."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(
            torch.empty(args.n_routed_experts, args.dim)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and expert indices."""
        logits = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            scores = logits.sigmoid()
        topk_scores, topk_indices = scores.topk(
            self.n_activated_experts, dim=-1, sorted=False
        )
        if self.score_func == "sigmoid":
            topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)
        topk_scores = topk_scores * self.route_scale
        return topk_scores.type_as(x), topk_indices


class FeedForward(nn.Module):
    """Standard SwiGLU feed-forward network for dense layers."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
