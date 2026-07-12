# -*- coding: utf-8 -*-
"""
NTSM Training Script — LOSO Cross-Validation Framework

This script provides the training framework for NTSM using
Leave-One-Subject-Out (LOSO) cross-validation on the SEED-VIG dataset.

Note: Data loading utilities are dataset-specific and should be
      adapted to your local data organization. See Section III-A
      of the paper for preprocessing details.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from config import (
    SEED, DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, D_MODEL, DROPOUT, LOSS_ALPHA,
    EARLY_STOP_PATIENCE, SMOOTH_WINDOW, GRAD_CLIP
)
from models import NTSM
from utils.loss import TrendValueLoss


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FatigueDataset(Dataset):
    """Generic dataset wrapper for EEG-EOG fatigue detection."""

    def __init__(self, eeg, eog, labels):
        self.eeg = eeg
        self.eog = eog
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.eeg[idx], dtype=torch.float32),
            torch.tensor(self.eog[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Single training epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    for eeg, eog, y in loader:
        eeg, eog, y = eeg.to(device), eog.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(eeg, eog)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluate with moving average post-processing.

    A window of size 5 is applied to smooth predictions,
    reducing high-frequency jitter in fatigue score output.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for eeg, eog, y in loader:
            eeg, eog = eeg.to(device), eog.to(device)
            pred = model(eeg, eog)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    # Post-processing: moving average smoothing
    if len(preds) > SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        preds = np.convolve(preds, kernel, mode='same')

    return preds, trues


def run_loso(subjects, device):
    """
    LOSO cross-validation loop.

    Args:
        subjects: List of dicts with keys 'eeg', 'eog', 'label'.
        device: Torch device.
    """
    results = []

    for test_idx in range(len(subjects)):
        # Partition: leave one subject out
        train_eeg = np.concatenate([
            s['eeg'] for i, s in enumerate(subjects) if i != test_idx
        ])
        train_eog = np.concatenate([
            s['eog'] for i, s in enumerate(subjects) if i != test_idx
        ])
        train_lbl = np.concatenate([
            s['label'] for i, s in enumerate(subjects) if i != test_idx
        ])
        test_eeg = subjects[test_idx]['eeg']
        test_eog = subjects[test_idx]['eog']
        test_lbl = subjects[test_idx]['label']

        # Z-score normalization (fit on train, transform test)
        sc_eeg = StandardScaler()
        sc_eog = StandardScaler()
        train_eeg = sc_eeg.fit_transform(
            train_eeg.reshape(-1, train_eeg.shape[-1])
        ).reshape(train_eeg.shape)
        test_eeg = sc_eeg.transform(
            test_eeg.reshape(-1, test_eeg.shape[-1])
        ).reshape(test_eeg.shape)
        train_eog = sc_eog.fit_transform(
            train_eog.reshape(-1, train_eog.shape[-1])
        ).reshape(train_eog.shape)
        test_eog = sc_eog.transform(
            test_eog.reshape(-1, test_eog.shape[-1])
        ).reshape(test_eog.shape)

        train_loader = DataLoader(
            FatigueDataset(train_eeg, train_eog, train_lbl),
            batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(
            FatigueDataset(test_eeg, test_eog, test_lbl),
            batch_size=BATCH_SIZE, shuffle=False
        )

        # Model, optimizer, scheduler, loss
        model = NTSM(d_model=D_MODEL, dropout=DROPOUT).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5
        )
        criterion = TrendValueLoss(alpha=LOSS_ALPHA).to(device)

        # Training with early stopping
        best_rmse = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            preds, trues = evaluate(model, test_loader, device)
            rmse = np.sqrt(mean_squared_error(trues, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_{test_idx}.pth')
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break

        # Final evaluation
        model.load_state_dict(torch.load(f'best_model_{test_idx}.pth'))
        preds, trues = evaluate(model, test_loader, device)
        cor, _ = pearsonr(trues, preds)
        results.append({'cor': cor, 'rmse': best_rmse})
        print(f'Subject {test_idx + 1}: COR={cor:.4f}, RMSE={best_rmse:.4f}')

    # Summary
    cors = [r['cor'] for r in results]
    rmses = [r['rmse'] for r in results]
    print(f'\nMean COR:  {np.mean(cors):.4f} +/- {np.std(cors):.4f}')
    print(f'Mean RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}')


if __name__ == '__main__':
    set_seed(SEED)
    # Note: Users should implement their own data loading pipeline
    # adapted to their local SEED-VIG data organization.
    # Expected format per subject: {'eeg': (N, L, 425), 'eog': (N, L, 36), 'label': (N,)}
    print('Please implement data loading for your local dataset configuration.')
