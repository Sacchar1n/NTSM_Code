# -*- coding: utf-8 -*-
"""
Trend-Value Hybrid Loss (TV-Loss)

Paper Reference: Eq. 16
    L = α · L_SmoothL1 + (1 - α) · (1 - COR)

where:
    - L_SmoothL1: Smooth L1 loss for numerical accuracy
    - COR: Pearson Correlation Coefficient for trend consistency
    - α = 0.6 (default, as specified in Table I)

This combined loss ensures the model not only predicts accurate numerical
values but also captures the continuous evolution trend of fatigue.
"""

import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    """
    Differentiable Pearson Correlation Loss.

    Computes 1 - COR(preds, targets), which equals 0 when predictions
    perfectly correlate with targets and approaches 2 for perfect
    anti-correlation.

    Input:
        preds: Predicted values, shape (batch,) or (batch, 1).
        targets: Ground-truth values, shape (batch,) or (batch, 1).

    Output:
        Scalar loss value in [0, 2].
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        # Center the variables
        vx = preds - torch.mean(preds)
        vy = targets - torch.mean(targets)
        # Pearson correlation
        cost = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2))
            * torch.sqrt(torch.sum(vy ** 2))
            + 1e-8
        )
        return 1 - cost


class TrendValueLoss(nn.Module):
    """
    Trend-Value Hybrid Loss (TV-Loss).

    Combines SmoothL1 loss (numerical accuracy) with Pearson correlation
    loss (trend consistency) for fatigue score prediction.

    L = α · L_SmoothL1 + (1 - α) · (1 - COR)

    Args:
        alpha (float): Weight for SmoothL1 component. Default: 0.6.
                       Higher α emphasizes numerical accuracy.
                       Lower α emphasizes trend consistency.

    Input:
        preds: Predicted PERCLOS scores, shape (batch,).
        targets: Ground-truth PERCLOS scores, shape (batch,).

    Output:
        Scalar combined loss.
    """

    def __init__(self, alpha=0.6):
        super().__init__()
        self.mse_loss = nn.SmoothL1Loss()
        self.corr_loss = PearsonLoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        return (
            self.alpha * self.mse_loss(preds, targets)
            + (1 - self.alpha) * self.corr_loss(preds, targets)
        )
