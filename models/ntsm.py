# -*- coding: utf-8 -*-
"""
NTSM: Neurophysiology-Guided Tri-Stream Selective State Space Model

Paper Reference: Section II (Full Architecture)
This module assembles all neurophysiological interaction modules into
the complete NTSM architecture for driver fatigue estimation.

Architecture Overview (Fig. 1):
    ┌─────────────────────────────────────────────────────────────┐
    │ Input: EEG (17ch × 25 sub-bands = 425-dim) + EOG (36-dim)  │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
      T-Stream (150d)     P-Stream (275d)     EOG-Stream (36d)
         [SRU]               [SRU]           [BilinearEncoder]
              │                   │                   │
         [AD-Gate] ◄──────────────┼───────────────────┤
              │                   │                   │
       [Bi-SSM Encoder]   [Bi-SSM Encoder]   [Bi-SSM Encoder]
              │                   │                   │
              │◄── [FC-Bridge] ◄──┘                   │
              │                   │                   │
          [OGA] ◄─────────────────┼───────────────────┤
              │               [OGA] ◄─────────────────┤
              │                   │                   │
              └───────┬───────────┘───────────────────┘
                      │
             [Gated Fusion + Regression]
                      │
                      ▼
                 PERCLOS (0~1)
"""

import torch
import torch.nn as nn

from .sru import SpectralRecalibrationUnit
from .ad_gate import ArtifactDisentanglementGate
from .fc_bridge import FunctionalConnectivityBridge
from .oga import OcularGuidedAttention
from .ssm_encoder import BidirectionalSSMEncoder
from .bilinear_encoder import BilinearOcularEncoder
from .coherence_extractor import CoherenceExtractor


class NTSM(nn.Module):
    """
    NTSM: Neurophysiology-Guided Tri-Stream Selective State Space Model.

    Integrates Mamba-based selective SSM with four neurophysiological
    interaction modules for real-time driver fatigue detection.

    Args:
        d_model (int): Model dimension D. Default: 64.
        num_layers (int): Number of Bi-SSM layers per stream. Default: 1.
        dropout (float): Dropout rate. Default: 0.25.
        t_dim (int): Temporal stream input dimension. Default: 150.
        p_dim (int): Parietal stream input dimension. Default: 275.
        eog_dim (int): EOG feature dimension. Default: 36.
        num_heads (int): Number of attention heads in OGA. Default: 4.

    Input:
        eeg: EEG-DE features of shape (batch, seq_len, 425).
             First 150 dims = Temporal region, last 275 dims = Parietal region.
        eog: EOG features of shape (batch, seq_len, 36).

    Output:
        Predicted PERCLOS score of shape (batch,).
    """

    def __init__(self, d_model=64, num_layers=1, dropout=0.25,
                 t_dim=150, p_dim=275, eog_dim=36, num_heads=4):
        super().__init__()

        # ========== 1. Input Dimensions ==========
        self.t_dim = t_dim
        self.p_dim = p_dim
        self.eog_dim = eog_dim

        # ========== 2. Embedding Layers (with SRU) ==========
        # Temporal stream: Linear projection + SRU for spectral recalibration
        self.embed_t = nn.Sequential(
            nn.Linear(self.t_dim, d_model),
            SpectralRecalibrationUnit(d_model)
        )
        # Parietal stream: Linear projection + SRU for spectral recalibration
        self.embed_p = nn.Sequential(
            nn.Linear(self.p_dim, d_model),
            SpectralRecalibrationUnit(d_model)
        )
        # Ocular stream: Bilinear encoding for EOG features
        self.embed_eog = BilinearOcularEncoder(self.eog_dim, d_model)

        # ========== 3. Artifact Disentanglement (AD-Gate) ==========
        # Applied to temporal stream to remove EOG artifacts
        self.ad_gate = ArtifactDisentanglementGate(d_model)

        # ========== 4. Temporal Encoders (Bidirectional SSM) ==========
        # Each stream has its own Bi-SSM encoder with LayerNorm
        self.encoder_t = nn.Sequential(
            nn.LayerNorm(d_model),
            *[BidirectionalSSMEncoder(d_model) for _ in range(num_layers)]
        )
        self.encoder_p = nn.Sequential(
            nn.LayerNorm(d_model),
            *[BidirectionalSSMEncoder(d_model) for _ in range(num_layers)]
        )
        self.encoder_eog = nn.Sequential(
            nn.LayerNorm(d_model),
            *[BidirectionalSSMEncoder(d_model) for _ in range(num_layers)]
        )

        # ========== 5. Functional Connectivity Bridge (FC-Bridge) ==========
        # Parietal stream modulates Temporal stream
        self.fc_bridge = FunctionalConnectivityBridge(d_model)

        # ========== 6. Ocular-Guided Attention (OGA) ==========
        # Applied to both T and P streams
        self.oga_t = OcularGuidedAttention(d_model, num_heads=num_heads, dropout=dropout)
        self.oga_p = OcularGuidedAttention(d_model, num_heads=num_heads, dropout=dropout)

        # ========== 7. Coherence Extractor ==========
        # Cross-modal synchrony feature (D/4 = 16 dims)
        self.coherence_extractor = CoherenceExtractor(d_model)

        # ========== 8. Gated Fusion Network (Section II-E) ==========
        # Input: [H_T' || H_P || H_eog || H_sync] = 3D + D/4
        fusion_dim = d_model * 3 + d_model // 4
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Softmax(dim=1)
        )

        # ========== 9. Regression Head ==========
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(fusion_dim, 1)

    def forward(self, eeg, eog):
        """
        Forward pass of NTSM.

        Args:
            eeg: EEG-DE features, shape (batch, seq_len, t_dim + p_dim).
            eog: EOG features, shape (batch, seq_len, eog_dim).

        Returns:
            Predicted PERCLOS scores, shape (batch,).
        """
        # --- Slice EEG-DE features into T and P regions ---
        # Temporal (T): 6 channels × 25 sub-bands = 150 dims
        x_t_in = eeg[:, :, :self.t_dim]
        # Parietal-Occipital (P): 11 channels × 25 sub-bands = 275 dims
        x_p_in = eeg[:, :, self.t_dim:]

        # --- Stage 1: Embedding ---
        h_t = self.embed_t(x_t_in)       # (B, L, D)
        h_p = self.embed_p(x_p_in)       # (B, L, D)
        h_eog = self.embed_eog(eog)      # (B, L, D)

        # --- Stage 2: Artifact Disentanglement ---
        # AD-Gate: clean temporal EEG using EOG guidance
        h_t = self.ad_gate(h_t, h_eog)

        # --- Stage 3: Bidirectional SSM Encoding ---
        h_t = self.encoder_t(h_t)        # (B, L, D)
        h_p = self.encoder_p(h_p)        # (B, L, D)
        h_eog = self.encoder_eog(h_eog)  # (B, L, D)

        # --- Stage 4: Functional Connectivity (P → T) ---
        # Parietal stream modulates Temporal stream
        h_t = self.fc_bridge(feat_p=h_p, feat_t=h_t)

        # --- Stage 5: Ocular-Guided Attention ---
        # EOG guides both T and P streams
        h_t_enhanced = self.oga_t(h_t, h_eog)
        h_p_enhanced = self.oga_p(h_p, h_eog)

        # --- Stage 6: Coherence Extraction ---
        feat_sync = self.coherence_extractor(h_p, h_eog)  # (B, D/4)

        # --- Stage 7: Global Temporal Pooling ---
        feat_t = h_t_enhanced.mean(dim=1)     # (B, D)
        feat_p = h_p_enhanced.mean(dim=1)     # (B, D)
        feat_eog = h_eog.mean(dim=1)          # (B, D)

        # --- Stage 8: Gated Fusion (Eqs. 13-15) ---
        # Concatenate all features
        concat = torch.cat([feat_t, feat_p, feat_eog, feat_sync], dim=1)
        # Generate gate weights (Eq. 14)
        w = self.gate(concat)  # (B, 4), softmax normalized

        # Weighted fusion (Eq. 15)
        final_feat = torch.cat([
            feat_t * w[:, 0:1],
            feat_p * w[:, 1:2],
            feat_eog * w[:, 2:3],
            feat_sync * w[:, 3:4]
        ], dim=1)

        # --- Stage 9: Regression ---
        return self.regressor(self.dropout(final_feat)).squeeze(-1)
