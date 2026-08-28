from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    alpha_factor = alpha * target + (1.0 - alpha) * (1.0 - target)
    modulating_factor = torch.pow(1.0 - p_t, gamma)
    loss = alpha_factor * modulating_factor * bce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def build_loss(loss_name: str, focal_alpha: float, focal_gamma: float, pos_weight: float | None):
    normalized_name = str(loss_name).lower()
    if normalized_name == "focal":
        return lambda logits, target: binary_focal_loss_with_logits(
            logits,
            target,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    if normalized_name == "bce":
        weight = None if pos_weight is None else torch.tensor(float(pos_weight))

        def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            local_weight = None
            if weight is not None:
                local_weight = weight.to(device=logits.device, dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=local_weight)

        return _loss
    raise ValueError(f"Unsupported loss_name={loss_name!r}. Expected 'focal' or 'bce'.")