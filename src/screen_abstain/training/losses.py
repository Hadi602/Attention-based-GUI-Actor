"\"\"\"
Loss helpers for training the null-patch action head.

The KL divergence over `n_enc + 1` patches lives inside
`VisionHead_MultiPatchWithNull.forward`. This module just provides:

    * sample-level weighting between positive and refusal examples
    * a regularisation term that discourages the null-patch logit from
      becoming too dominant on positive samples (helps avoid trivial
      \"always refuse\" solutions early in training)
\"\"\"

from __future__ import annotations

import torch


def positive_negative_weighted_loss(
    losses_per_sample: torch.Tensor,    # (B,)
    refusal_labels:    torch.Tensor,    # (B,) bool / 0-1
    pos_weight: float = 1.0,
    refusal_weight: float = 1.0,
) -> torch.Tensor:
    \"\"\"Weighted mean over a batch.\"\"\"
    refusal_mask = refusal_labels.bool().float()
    pos_mask = 1.0 - refusal_mask
    weights = pos_weight * pos_mask + refusal_weight * refusal_mask
    return (losses_per_sample * weights).sum() / weights.sum().clamp(min=1.0)


def null_suppression_regulariser(
    attn_weights: torch.Tensor,     # (B, n_enc + 1)
    refusal_labels: torch.Tensor,   # (B,)
    eps: float = 1e-6,
) -> torch.Tensor:
    \"\"\"
    Penalise high P(null) on *positive* samples.

    L_reg = - mean over positive samples of log(1 - P(null)).
    Encourages the model to keep the null-patch probability low when
    a real target exists. Multiply by a small coefficient (e.g. 0.05)
    when adding to the main loss.
    \"\"\"
    p_null = attn_weights[..., -1]     # (B,)
    pos_mask = (~refusal_labels.bool()).float()
    if pos_mask.sum() < 1:
        return attn_weights.new_tensor(0.0)
    log_term = torch.log(1.0 - p_null + eps)
    return -(log_term * pos_mask).sum() / pos_mask.sum()
"
