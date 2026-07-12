# -*- coding: utf-8 -*-
"""
Bidirectional Selective State Space Model (Bi-SSM) Encoder

Paper Reference: Section II-D
Core Innovation:
    Replaces Transformer encoders with bidirectional selective SSM to reduce
    computational complexity from quadratic O(L^2) to linear O(L).

Components:
    1. SSMUnit: A single selective state space model block based on Mamba.
    2. BidirectionalSSMEncoder: Combines forward and backward SSMUnits to
       capture both historical accumulation and future trend cues.

SSM Preliminaries (Eqs. 3-6):
    Continuous: h'(t) = A·h(t) + B·x(t),  y(t) = C·h(t)
    Discrete (ZOH): A_bar = exp(Δ·A),  B_bar = (ΔA)^{-1}(exp(ΔA) - I)·ΔB
    Recursive: h_t = A_bar · h_{t-1} + B_bar · x_t

Selective Scan (Eqs. 7-8):
    B_t = s_B(x_t),  C_t = s_C(x_t),  Δ_t = Softplus(Linear(x_t))
    This makes parameters input-dependent for adaptive filtering.

Bidirectional Architecture (Eq. 12):
    H_context = Linear([h_fwd || h_bwd]) + x  (with residual connection)
    Forward and backward encoders use independent (non-shared) parameters.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Selective Scan Implementation
# ============================================================

# Global switch: use JIT by default (fast), disable temporarily
# for gradient saliency computation (requires autograd compatibility)
_USE_JIT_SCAN = True


@torch.jit.script
def selective_scan_core(
    delta_A: torch.Tensor, delta_B_u: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core loop of the selective scan algorithm.

    Args:
        delta_A: Discretized state transition, shape (B, L, D, N).
        delta_B_u: Discretized input contribution, shape (B, L, D, N).
        C: Output projection, shape (B, L, N).

    Returns:
        Output sequence y, shape (B, L, D).
    """
    B_batch, L, D_model, N = delta_A.shape
    h = torch.zeros(
        (B_batch, D_model, N), device=delta_A.device, dtype=delta_A.dtype
    )
    ys: List[torch.Tensor] = []

    for t in range(L):
        h = delta_A[:, t, :, :] * h + delta_B_u[:, t, :, :]
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)

    return torch.stack(ys, dim=1)


def selective_scan(u, delta, A, B, C, D):
    """
    Full selective scan with discretization.

    Implements the core SSM computation:
        1. Discretize A, B using zero-order hold with input-dependent Δ.
        2. Run the sequential state update (JIT or pure Python).
        3. Add skip connection via D parameter.

    Args:
        u: Input sequence, shape (B, L, D_inner).
        delta: Time step Δ, shape (B, L, D_inner).
        A: State transition (log-space), shape (D_inner, N).
        B: Input projection, shape (B, L, N).
        C: Output projection, shape (B, L, N).
        D: Skip connection parameter, shape (D_inner,).

    Returns:
        Output sequence y, shape (B, L, D_inner).
    """
    # Discretize: delta_A = exp(Δ · A)
    delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    # Discretize: delta_B_u = Δ · B · u
    delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

    if _USE_JIT_SCAN:
        # JIT-compiled core loop (fast, for training/inference)
        y = selective_scan_core(delta_A, delta_B_u, C)
    else:
        # Pure Python loop (for gradient saliency, autograd compatible)
        B_batch, L, D_model, N = delta_A.shape
        h = torch.zeros(
            (B_batch, D_model, N), device=u.device, dtype=u.dtype
        )
        ys = []
        for t in range(L):
            h = delta_A[:, t, :, :] * h + delta_B_u[:, t, :, :]
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

    # Skip connection
    y = y + D.unsqueeze(0).unsqueeze(0) * u
    return y


def set_jit_scan(enabled):
    """
    Toggle JIT scan mode.

    Set to False before gradient saliency computation to ensure
    autograd compatibility, then restore to True afterwards.
    """
    global _USE_JIT_SCAN
    _USE_JIT_SCAN = enabled


# ============================================================
# SSM Unit (Single Mamba Block)
# ============================================================

class SSMUnit(nn.Module):
    """
    Selective State Space Model Unit (Mamba Block).

    Internal structure (Fig. 3a in paper):
        Input -> Linear(E×D) -> [upper: Conv1d -> SiLU -> selective SSM]
                              -> [lower: SiLU (gating)]
                              -> element-wise multiply -> Linear(D) -> Output

    Args:
        d_model (int): Model dimension D. Default: 64.
        d_state (int): SSM state dimension N. Default: 16.
        d_conv (int): Conv1d kernel size K. Default: 4.
        expand (int): Expansion factor E. Default: 2.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16))

        # Input projection: D -> 2 * D_inner (split into x and z branches)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D convolution for local feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise convolution
            padding=d_conv - 1,
        )

        # Projection for SSM parameters (dt_rank + 2*d_state)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )
        # Time step projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameter: log-space parameterization ensures delta_A ∈ (0, 1)
        A = torch.arange(
            1, d_state + 1, dtype=torch.float32
        ).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D parameter: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection: D_inner -> D
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Initialize dt bias using inverse softplus
        self._init_dt_bias()

    def _init_dt_bias(self, dt_min=0.001, dt_max=0.1):
        """Initialize Δ bias via inverse softplus for numerical stability."""
        dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj.bias._no_weight_decay = True

    def forward(self, x):
        """
        Forward pass of a single Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        (b, l, d) = x.shape

        # Input projection and split into two branches
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Upper branch: Conv1d -> SiLU -> selective SSM
        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = self.conv1d(x_ssm)[:, :, :l]
        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = F.silu(x_ssm)

        # Generate input-dependent SSM parameters (Eqs. 7-8)
        x_dbl = self.x_proj(x_ssm)
        dt_raw, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Δ = Softplus(Linear(dt_raw)), ensuring Δ > 0
        delta = F.softplus(self.dt_proj(dt_raw))
        # A in negative log-space
        A = -torch.exp(self.A_log.float())

        # Core selective scan state recursion
        y = selective_scan(
            u=x_ssm, delta=delta,
            A=A, B=B_ssm, C=C_ssm, D=self.D,
        )

        # Gating: multiply with lower branch (SiLU activation)
        y = y * F.silu(z)
        return self.out_proj(y)


# ============================================================
# Bidirectional SSM Encoder
# ============================================================

class BidirectionalSSMEncoder(nn.Module):
    """
    Bidirectional Selective SSM Encoder (Fig. 3b in paper).

    Combines forward and backward SSMUnits with independent (non-shared)
    parameters to capture both historical accumulation and future trends.

    Architecture (Eq. 12):
        H_context = Linear([h_fwd || h_bwd]) + x  (residual connection)

    Args:
        d_model (int): Model dimension D. Default: 64.
        d_state (int): SSM state dimension N. Default: 16.
        d_conv (int): Conv1d kernel size K. Default: 4.
        expand (int): Expansion factor E. Default: 2.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Forward and backward SSMs with independent parameters
        self.fwd_ssm = SSMUnit(d_model, d_state, d_conv, expand)
        self.bwd_ssm = SSMUnit(d_model, d_state, d_conv, expand)
        # Linear fusion: 2D -> D
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass.

        Steps:
            1. Forward SSM: process sequence in chronological order.
            2. Backward SSM: process reversed sequence.
            3. Concatenate and fuse with linear projection.
            4. Add residual connection.
        """
        # Forward pathway: captures cumulative fatigue effects
        out_fwd = self.fwd_ssm(x)
        # Backward pathway: captures future contextual information
        x_rev = torch.flip(x, dims=[1])
        out_bwd = self.bwd_ssm(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        # Concatenate and fuse
        combined = torch.cat([out_fwd, out_bwd], dim=-1)
        # Linear fusion + residual connection (Eq. 12)
        return self.norm(self.fusion(combined)) + x
