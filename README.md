# NTSM

> **Neurophysiology-Guided Tri-Stream Selective State Space Model for Real-Time Driver Fatigue Detection with EEG-EOG Signals**  
> IEEE Journal of Biomedical and Health Informatics (J-BHI)

## Overview

Official PyTorch implementation of **NTSM**, a neurophysiology-guided tri-stream architecture integrating the Mamba-based selective State Space Model (SSM) with four specialized neurophysiological interaction modules for EEG-EOG fatigue detection.

## Architecture

NTSM consists of three parallel processing streams (Temporal, Parietal, Ocular) connected by four interaction modules:

| Module | Description | Paper Reference |
|--------|-------------|-----------------|
| **FC-Bridge** | Functional Connectivity Bridge — parietal-to-temporal top-down attentional regulation | Section II-C1 |
| **SRU** | Spectral Recalibration Unit — channel-wise attention for adaptive frequency band weighting | Section II-C2 |
| **OGA** | Ocular-Guided Attention — asymmetric cross-attention for eye-brain coupling | Section II-C3 |
| **AD-Gate** | Artifact Disentanglement Gate — soft disentanglement of EOG artifacts | Section II-C4 |
| **Bi-SSM** | Bidirectional Selective SSM Encoder — linear-complexity temporal modeling | Section II-D |

## Repository Structure

```
NTSM_Code/
├── config.py                   # Hyperparameter configuration (Table I)
├── models/
│   ├── ntsm.py                 # Main NTSM architecture
│   ├── sru.py                  # Spectral Recalibration Unit
│   ├── fc_bridge.py            # Functional Connectivity Bridge
│   ├── oga.py                  # Ocular-Guided Attention
│   ├── ad_gate.py              # Artifact Disentanglement Gate
│   ├── ssm_encoder.py          # SSM Unit & Bidirectional SSM Encoder
│   ├── bilinear_encoder.py     # Bilinear Ocular Encoder
│   └── coherence_extractor.py  # Cross-modal Coherence Extractor
├── utils/
│   └── loss.py                 # Trend-Value Hybrid Loss (Eq. 16)
├── train.py                    # Training example (LOSO protocol)
├── requirements.txt
└── LICENSE
```

## Requirements

- Python >= 3.9
- PyTorch >= 1.9
- NumPy, SciPy, scikit-learn

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from models import NTSM

model = NTSM(d_model=64)

# Input: EEG-DE features (B, L, 425) + EOG features (B, L, 36)
eeg = torch.randn(8, 16, 425)
eog = torch.randn(8, 16, 36)

output = model(eeg, eog)  # (B,) continuous PERCLOS scores
```

## Hyperparameters (Table I)

| Category | Parameter | Value |
|----------|-----------|-------|
| Training | Epochs / Batch / Sequence Length | 60 / 64 / 16 |
| Training | Optimizer | AdamW (lr=1e-3) |
| Training | Scheduler | CosineAnnealingLR |
| Model | Dimension D / State N | 64 / 16 |
| Model | Dropout / Gradient Clip | 0.4 / 1.0 |
| Loss | Weight alpha | 0.6 |
| Loss | Early Stop Patience | 15 epochs |

## Citation

```bibtex
@article{yuan2026ntsm,
  title={NTSM: Neurophysiology-Guided Tri-Stream Selective State Space Model for Real-Time Driver Fatigue Detection with EEG-EOG Signals},
  author={Yuan, Jiantao and Tang, Jing and Han, Haijun and Yin, Rui and Liu, Shengli and Wang, Jue and Wu, Celimuge},
  journal={IEEE Journal of Biomedical and Health Informatics},
  note={Under Review},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
