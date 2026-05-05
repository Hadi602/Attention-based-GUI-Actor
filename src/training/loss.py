"\"\"\"
ScreenAbstain dual-task loss.

Total loss = grounding loss (on groundable samples)
           + lambda_ref * refusal loss (on refusal samples)
           + lambda_marg * margin penalty (optional)

Grounding loss: cross-entropy over (N+1) categories where the target
is the gold patch index. The null index is a *valid* class but never
the target on groundable samples — so the optimizer naturally pushes
mass off `null_patch` for these.

Refusal loss: cross-entropy where the target is the null index (N).
Symmetric, simple, and compatible with the same softmax.

Margin penalty (optional): hinge that enforces
    p_refuse_groundable < threshold_low
    p_refuse_refusable  > threshold_high
helping calibrate the threshold without per-task tuning.

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ScreenAbstainLossConfig:
    lambda_ref: float = 1.0
    lambda_margin: float = 0.0
    margin_low: float = 0.1     # groundable samples should have p_refuse < this
    margin_high: float = 0.7    # refusal samples should have p_refuse > this
    label_smoothing: float = 0.0


class ScreenAbstainLoss(nn.Module):
    \"\"\"
    Inputs to forward():
        attn_logits : (B, N+1)
        target_idx  : (B,) long  — gold patch index for groundable,
                                   -1 for refusal samples
        is_refusal  : (B,) bool  — True if sample is a refusal example
    \"\"\"

    def __init__(self, cfg: Optional[ScreenAbstainLossConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ScreenAbstainLossConfig()

    def forward(
        self,
        attn_logits: torch.Tensor,
        target_idx: torch.Tensor,
        is_refusal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, K = attn_logits.shape
        null_idx = K - 1
        device = attn_logits.device

        # Build per-sample target: refusal → null_idx, else original.
        targets = torch.where(
            is_refusal,
            torch.full_like(target_idx, null_idx),
            target_idx,
        )

        # Sanity: groundable targets must be valid patch indices.
        # (Caller is responsible; we just clamp for safety.)
        targets = targets.clamp(min=0, max=null_idx)

        # Per-sample CE (no reduction)
        ce = F.cross_entropy(
            attn_logits,
            targets,
            reduction=\"none\",
            label_smoothing=self.cfg.label_smoothing,
        )  # (B,)

        ground_mask = ~is_refusal
        refuse_mask = is_refusal

        n_ground = ground_mask.sum().clamp_min(1)
        n_refuse = refuse_mask.sum().clamp_min(1)

        loss_ground = (ce * ground_mask).sum() / n_ground
        loss_refuse = (ce * refuse_mask).sum() / n_refuse

        loss = loss_ground + self.cfg.lambda_ref * loss_refuse

        # Optional margin / hinge penalty on p_refuse
        margin_loss = torch.tensor(0.0, device=device)
        if self.cfg.lambda_margin > 0:
            probs = F.softmax(attn_logits, dim=-1)
            p_ref = probs[:, null_idx]                      # (B,)
            # Groundable: penalize p_ref above margin_low
            up = F.relu(p_ref - self.cfg.margin_low) * ground_mask
            # Refusable: penalize p_ref below margin_high
            down = F.relu(self.cfg.margin_high - p_ref) * refuse_mask
            margin_loss = up.sum() / n_ground + down.sum() / n_refuse
            loss = loss + self.cfg.lambda_margin * margin_loss

        return {
            \"loss\": loss,
            \"loss_ground\": loss_ground.detach(),
            \"loss_refuse\": loss_refuse.detach(),
            \"loss_margin\": margin_loss.detach(),
            \"n_ground\": n_ground.detach(),
            \"n_refuse\": n_refuse.detach(),
        }
"
