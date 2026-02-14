import torch
from torch import nn

class QuantileLoss(nn.Module):
    """
    Pinball loss (Quantile Regression Loss) for multiple quantiles.

    Args:
        quantiles (list of float): List of quantiles to estimate, e.g. [0.05, 0.5, 0.95].

    Input:
        preds:  (B, Q, H, W) predicted quantile maps
        target: (B, 1, H, W) true values

    Output:
        loss (torch.Tensor): Scalar loss value
    """
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1, :, :]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())

        return sum(losses) / len(losses)