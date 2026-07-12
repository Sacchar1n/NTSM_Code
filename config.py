# -*- coding: utf-8 -*-
"""
Configuration for NTSM (Table I in the paper).

All hyperparameters are defined here to ensure full reproducibility.
"""

import os
import torch

# ============================================================
# Random Seed
# ============================================================
SEED = 42

# ============================================================
# Device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SEED-VIG Dataset: 17-Channel Electrode Configuration
# ============================================================
# Channels are divided into two brain regions:
#   - Temporal (T-region): Channels 1-6  -> FT7, FT8, T7, T8, TP7, TP8
#   - Parietal-Occipital (P-region): Channels 7-17 -> CP1, CP2, P1, Pz, P2,
#                                                      PO3, POz, PO4, O1, Oz, O2
SEED_VIG_CHANNELS = [
    'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',       # Temporal lobe (1-6)
    'CP1', 'CP2', 'P1', 'Pz', 'P2',                 # Central-Parietal (7-11)
    'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2'           # Parieto-Occipital (12-17)
]

# ============================================================
# Training Hyperparameters (Table I)
# ============================================================
EPOCHS = 60                # Total training epochs
BATCH_SIZE = 64            # Mini-batch size
SEQ_LEN = 16               # Sliding window length (L = 16 frames)
LEARNING_RATE = 1e-3       # AdamW optimizer learning rate (eta)
WEIGHT_DECAY = 1e-2        # AdamW weight decay

# ============================================================
# Model Architecture (Table I)
# ============================================================
D_MODEL = 64               # Model dimension (D)
D_STATE = 16               # SSM state dimension (N)
D_CONV = 4                 # Conv1d kernel size (K)
EXPAND = 2                 # SSM expansion factor (E), inner dim = D * E = 128
NUM_LAYERS = 1             # Number of Bi-SSM encoder layers per stream
DROPOUT = 0.4              # Dropout rate
NUM_HEADS = 4              # Number of attention heads in OGA
GRAD_CLIP = 1.0            # Gradient clipping norm

# ============================================================
# Input Feature Dimensions
# ============================================================
# EEG: 17 channels x 25 sub-bands (2Hz bandwidth each, 0-50Hz)
#   - Temporal stream (T): 6 channels x 25 = 150 dims
#   - Parietal stream (P): 11 channels x 25 = 275 dims
# EOG: 36-dimensional statistical features
T_DIM = 150                # Temporal stream input dimension
P_DIM = 275                # Parietal stream input dimension
EOG_DIM = 36               # EOG feature dimension

# ============================================================
# Loss Function (Eq. 16 in the paper)
# ============================================================
# L = alpha * L_SmoothL1 + (1 - alpha) * (1 - COR)
LOSS_ALPHA = 0.6           # Weight for SmoothL1 loss component

# ============================================================
# Early Stopping
# ============================================================
EARLY_STOP_PATIENCE = 15   # Number of epochs to wait before stopping

# ============================================================
# Post-Processing
# ============================================================
SMOOTH_WINDOW = 5          # Moving average window size for prediction smoothing

# ============================================================
# Output Directory
# ============================================================
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "ntsm_results")
