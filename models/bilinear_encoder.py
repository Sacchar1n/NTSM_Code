# -*- coding: utf-8 -*-
"""
Bilinear Ocular Encoder

Encodes EOG features into the shared representation space using a
bilinear gating mechanism, producing rich cross-term interactions
suitable for downstream cross-modal fusion.

Design:
    h1 = Linear_1(x)       # Feature transformation
    h2 = σ(Linear_2(x))    # Gating signal
    out = Linear_out(h1 ⊙ h2)  # Gated bilinear interaction
"""

import torch.nn as nn


class BilinearOcularEncoder(nn.Module):
    """
    Bilinear Ocular Encoder for EOG feature embedding.

    Uses a gated bilinear mechanism to encode EOG features.

    Args:
        in_dim (int): Input EOG feature dimension (36 for SEED-VIG).
        out_dim (int): Output embedding dimension (d_model).

    Input:
        x: EOG features of shape (batch, seq_len, in_dim).

    Output:
        Encoded features of shape (batch, seq_len, out_dim).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.act = nn.Sigmoid()
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """
        Forward pass.

        Steps:
            1. Two parallel linear transformations.
            2. Sigmoid gating on one branch.
            3. Element-wise product (bilinear interaction).
            4. Output projection.
        """
        h1 = self.linear1(x)
        h2 = self.act(self.linear2(x))
        out = h1 * h2
        return self.out_proj(out)
