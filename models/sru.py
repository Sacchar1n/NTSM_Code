# -*- coding: utf-8 -*-
"""
Spectral Recalibration Unit (SRU)

Paper Reference: Section II-C2
Neurophysiological Basis:
    Fatigue is characterized by increased power in delta (1-4 Hz) and
    theta (4-8 Hz) bands, while high-frequency gamma bands often contain
    irrelevant electromyographic noise. SRU adaptively enhances fatigue-
    related slow-wave features and suppresses noise bands via a channel-wise
    attention mechanism inspired by Squeeze-and-Excitation networks.

Design Notes:
    - SRU is NOT a classical PSD estimator (e.g., Welch's method).
    - It is strictly a channel-wise attention mechanism that dynamically
      recalibrates pre-extracted spectral features based on fatigue relevance.
    - Global Average Pooling (GAP) creates an auxiliary spectral summary
      for attention computation without compressing temporal dynamics.
    - The attention weights are broadcast, so the transient temporal dynamics
      of the input sequence are fully preserved in the output.
"""

import torch.nn as nn


class SpectralRecalibrationUnit(nn.Module):
    """
    Spectral Recalibration Unit (SRU).

    Implements channel-wise attention to adaptively weight frequency bands
    based on their relevance to fatigue detection.

    Args:
        channel (int): Number of input channels (feature dimension).
        reduction (int): Reduction ratio for the bottleneck. Default: 4.

    Input:
        x: Tensor of shape (batch, seq_len, channel).

    Output:
        Tensor of shape (batch, seq_len, channel), recalibrated features.
    """

    def __init__(self, channel, reduction=4):
        super().__init__()
        # Global Average Pooling along temporal dimension
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Bottleneck FC layers: squeeze -> excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Steps:
            1. Permute to (batch, channel, seq_len) for pooling.
            2. GAP to extract spectral descriptor z ∈ R^C.
            3. Generate attention weights s = σ(W₂·δ(W₁·z)).
            4. Recalibrate: Y_SRU = X ⊙ s (broadcast over temporal dim).
        """
        b, t, c = x.size()
        # Permute to (batch, channel, seq_len) for AdaptiveAvgPool1d
        y = x.permute(0, 2, 1)
        # GAP: (batch, channel, 1) -> (batch, channel)
        y = self.avg_pool(y).view(b, c)
        # Bottleneck: generate attention weights (batch, 1, channel)
        y = self.fc(y).view(b, 1, c)
        # Recalibrate input features
        return x * y
