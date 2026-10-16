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
        alpha: float = 2.1,
        foreground_weight: float = 10.0,
    ):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.foreground_weight = foreground_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        delta = (pred_fp32 - target_fp32).abs()

        exponent = self.alpha - target_fp32
        theta_eps_pow = torch.pow(self.theta / self.epsilon, exponent)

        A = (
            self.omega
            * (1.0 / (1.0 + theta_eps_pow))
            * exponent
            * torch.pow(self.theta / self.epsilon, exponent - 1.0)
            * (1.0 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(1.0 + theta_eps_pow)

        smooth_loss = self.omega * torch.log(
            1.0 + torch.pow(delta / self.epsilon, exponent)
        )
        linear_loss = A * delta - C

        loss = torch.where(delta < self.theta, smooth_loss, linear_loss)

        if self.foreground_weight > 0.0:
            weight_map = 1.0 + self.foreground_weight * (target_fp32 > 0.05).float()
            loss = loss * weight_map

        return loss.mean()

