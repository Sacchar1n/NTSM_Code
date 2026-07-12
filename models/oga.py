# -*- coding: utf-8 -*-
"""
Ocular-Guided Attention (OGA)

Paper Reference: Section II-C3
Neurophysiological Basis:
    Visual cortex (occipital lobe) activity is highly synchronized with
    eye movements such as saccades and blinks. EOG signals directly reflect
    these eye movement states and can guide the model to focus on critical
    neural activities associated with fatigue.

Design:
    OGA adopts an asymmetric cross-attention mechanism where EOG dynamically
    guides EEG feature extraction:
    - Query (Q) = EEG features  (what to attend to)
    - Key (K) / Value (V) = EOG features  (guiding information)
    - Attention = softmax(Q·K^T / √d_k) · V

    Example: When EOG signals indicate eyelid closure (a typical fatigue
    behavior), OGA automatically enhances the weight of visual cortex-related
    EEG features, realizing physiologically meaningful cross-modal fusion.
"""

import torch.nn as nn


class OcularGuidedAttention(nn.Module):
    """
    Ocular-Guided Attention (OGA).

    Uses asymmetric cross-attention where EOG guides EEG feature extraction,
    modeling the eye-brain coupling mechanism.

    Args:
        d_model (int): Feature dimension.
        num_heads (int): Number of attention heads. Default: 4.
        dropout (float): Dropout rate. Default: 0.1.

    Input:
        eeg_feat: EEG features of shape (batch, seq_len, d_model).
        eog_feat: EOG features of shape (batch, seq_len, d_model).

    Output:
        Enhanced EEG features of shape (batch, seq_len, d_model).
    """

    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        # Multi-head cross-attention: Q=EEG, K=V=EOG
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Post-attention normalization and FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, eeg_feat, eog_feat):
        """
        Forward pass.

        Steps:
            1. Cross-attention: Q=EEG, K=V=EOG.
            2. Residual connection + LayerNorm.
            3. Feed-forward network + residual + LayerNorm.
        """
        # Asymmetric cross-attention: EEG queries, EOG provides context
        attn_out, _ = self.attn(
            query=eeg_feat,
            key=eog_feat,
            value=eog_feat
        )
        eeg_feat = self.norm1(eeg_feat + self.dropout(attn_out))
        # Feed-forward network
        ffn_out = self.ffn(eeg_feat)
        return self.norm2(eeg_feat + self.dropout(ffn_out))
