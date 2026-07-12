# -*- coding: utf-8 -*-
"""
Cross-Modal Coherence Extractor

Captures temporal synchronization patterns between EEG and EOG modalities
using bilinear interaction and temporal pooling. Produces a low-dimensional
auxiliary feature (D/4) for the gated fusion stage.
"""

import torch.nn as nn


class CoherenceExtractor(nn.Module):
    """
    Cross-Modal Coherence Extractor.

    Computes a synchrony feature between EEG and EOG streams that
    captures their temporal coordination patterns.

    Args:
        d_model (int): Feature dimension.

    Input:
        eeg_feat: EEG features of shape (batch, seq_len, d_model).
        eog_feat: EOG features of shape (batch, seq_len, d_model).

    Output:
        Synchrony feature of shape (batch, d_model // 4).
    """

    def __init__(self, d_model):
        super().__init__()
        # Bilinear interaction: (d_model, d_model) -> d_model // 2
        self.bilinear = nn.Bilinear(d_model, d_model, d_model // 2)
        # Temporal pooling
        self.temp_pool = nn.AdaptiveAvgPool1d(1)
        # Output projection: d_model // 2 -> d_model // 4
        self.out_proj = nn.Linear(d_model // 2, d_model // 4)

    def forward(self, eeg_feat, eog_feat):
        """
        Forward pass.

        Steps:
            1. Bilinear interaction between EEG and EOG.
            2. Temporal average pooling.
            3. Project to low-dimensional synchrony feature.
        """
        sync = self.bilinear(eeg_feat, eog_feat)
        pooled = self.temp_pool(sync.transpose(1, 2)).squeeze(-1)
        return self.out_proj(pooled)
