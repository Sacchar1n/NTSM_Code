# -*- coding: utf-8 -*-
"""
Functional Connectivity Bridge (FC-Bridge)

Paper Reference: Section II-C1
Neurophysiological Basis:
    The parietal lobe serves as the central hub of the Dorsal Attention
    Network, which regulates the temporal lobe's processing of sensory
    inputs (Corbetta & Shulman, 2002). During fatigue, this top-down
    functional connectivity weakens, leading to cognitive control decline
    (Lim & Dinges, 2010).

Design:
    FC-Bridge explicitly models the parietal-to-temporal top-down regulation:
    1. Generate gating signal from parietal features:
       G_attn = σ(W_gate · H_P + b_gate)
    2. Modulate temporal features via residual enhancement:
       H_T' = H_T ⊙ (1 + G_attn)

    When G_attn ≈ 1 (alert state): temporal features are significantly enhanced.
    When G_attn ≈ 0 (fatigue state): enhancement diminishes, simulating
    weakened functional connectivity.
"""

import torch.nn as nn


class FunctionalConnectivityBridge(nn.Module):
    """
    Functional Connectivity Bridge (FC-Bridge).

    Simulates the parietal-to-temporal top-down attentional regulation
    pathway documented in cognitive neuroscience.

    Args:
        d_model (int): Feature dimension.

    Input:
        feat_p: Parietal stream features of shape (batch, seq_len, d_model).
        feat_t: Temporal stream features of shape (batch, seq_len, d_model).

    Output:
        Modulated temporal features of shape (batch, seq_len, d_model).
    """

    def __init__(self, d_model):
        super().__init__()
        # Gate generator: H_P -> attention gate G_attn ∈ (0, 1)
        self.gate_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat_p, feat_t):
        """
        Forward pass.

        Steps:
            1. Generate attention gate from parietal features.
            2. Modulate temporal features: H_T' = H_T + H_T * G_attn.
            3. Apply LayerNorm for training stability.
        """
        # Parietal (P) modulates Temporal (T)
        attention_gate = self.gate_generator(feat_p)
        feat_t_modulated = feat_t * attention_gate
        out = feat_t + feat_t_modulated
        return self.norm(out)
