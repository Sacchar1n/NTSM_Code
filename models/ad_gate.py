# -*- coding: utf-8 -*-
"""
Artifact Disentanglement Gate (AD-Gate)

Paper Reference: Section II-C4
Neurophysiological Basis:
    EEG and EOG signals are often mixed due to volume conduction. EOG
    signals contain both harmful potential artifacts that contaminate EEG
    and useful fatigue-related information (e.g., blink frequency, saccade
    velocity). Traditional hard-removal methods (e.g., ICA) may discard
    valid information while removing artifacts.

Design:
    AD-Gate adopts a soft-disentanglement strategy using an additive
    compensation mechanism:
    1. Project EOG features to EEG space: L_comp = W_proj · H_eog
    2. Generate gating signal: g = σ(W_g · [H_eeg ∥ L_comp])
    3. Additive calibration: H_clean = LayerNorm(H_eeg + α · g ⊙ L_comp)

    During training, W_proj spontaneously learns negative weights to subtract
    artifacts or positive weights to introduce useful fatigue information,
    unifying denoising and cross-modal complementation.
"""

import torch
import torch.nn as nn


class ArtifactDisentanglementGate(nn.Module):
    """
    Artifact Disentanglement Gate (AD-Gate).

    Softly disentangles EOG artifacts from EEG while preserving useful
    cross-modal fatigue information.

    Args:
        d_model (int): Feature dimension.
        alpha (float): Learnable scale for additive calibration. Default: 0.1.

    Input:
        eeg_feat: EEG features of shape (batch, seq_len, d_model).
        eog_feat: EOG features of shape (batch, seq_len, d_model).

    Output:
        Cleaned EEG features of shape (batch, seq_len, d_model).
    """

    def __init__(self, d_model, alpha=0.1):
        super().__init__()
        # Gating network: [H_eeg || L_comp] -> gate signal
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        # Projection: EOG -> EEG feature space
        self.proj = nn.Linear(d_model, d_model)
        # Layer normalization for output stability
        self.norm = nn.LayerNorm(d_model)
        # Scale factor for additive calibration
        self.alpha = alpha

    def forward(self, eeg_feat, eog_feat):
        """
        Forward pass.

        Steps:
            1. Project EOG to EEG space to get latent component L_comp.
            2. Concatenate [H_eeg || L_comp] and generate gating signal g.
            3. Additive calibration: H_clean = LN(H_eeg + α · g ⊙ L_comp).
        """
        # Step 1: Project EOG features to EEG space
        leakage = self.proj(eog_feat)
        # Step 2: Generate adaptive gating signal
        concat = torch.cat([eeg_feat, leakage], dim=-1)
        g = self.gate(concat)
        # Step 3: Additive calibration with learned scale
        out = eeg_feat + self.alpha * g * leakage
        return self.norm(out)
