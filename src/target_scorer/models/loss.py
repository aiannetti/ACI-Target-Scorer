"""
Adaptive Wing Loss for high-precision heatmap regression.
"""

import torch
import torch.nn as nn


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for robust heatmap regression.
    Behaves like dynamic exponential penalty for small sub-pixel peak residuals,
    and transitions to smooth linear loss for large errors.
    """
    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1
    ):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (pred - target).abs()
        A = self.omega * (
            1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1.0) * (1.0 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target)
        )

        smooth_loss = self.omega * torch.log(
            1.0 + torch.pow(delta / self.epsilon, self.alpha - target)
        )
        linear_loss = A * delta - C

        loss = torch.where(delta < self.theta, smooth_loss, linear_loss)
        return loss.mean()
