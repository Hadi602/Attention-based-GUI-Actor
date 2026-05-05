"\"\"\"
Null-Patch Action Head for GUI-Actor.

This is the core scientific contribution of ScreenAbstain.

We extend GUI-Actor's `VisionHead_MultiPatch` (Wu et al., NeurIPS 2025) by
appending a single learnable \"null patch\" embedding to the encoder side of the
attention computation. After concatenation the action head's softmax operates
over `n_enc + 1` items where index `n_enc` (the last) represents
\"no target on screen\" (refusal).

When training on a refusal sample, the supervisory target distribution places
all probability mass on the null patch. When training on a normal grounding
sample, the target distribution is the standard multi-patch label
(zero on the null index).

At inference time:
    abstain  iff  argmax(attn) == n_enc        # hard rule, default
or  abstain  iff  attn[..., n_enc] > tau       # soft thresholded rule

This module is a drop-in replacement for the original
`gui_actor.modeling.VisionHead_MultiPatch`.

Reference for the original head:
    https://github.com/microsoft/GUI-Actor/blob/main/src/gui_actor/modeling.py
\"\"\"

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VisionHead_MultiPatchWithNull(nn.Module):
    \"\"\"
    Drop-in replacement for `VisionHead_MultiPatch` that supports refusal.

    Compared to the original, the only architectural change is one learnable
    parameter `self.null_patch` of shape `(1, d_model)` that is concatenated to
    the encoder side of the dot-product attention.

    Args:
        d_model: hidden dim of the VLM (e.g. 3584 for Qwen2-VL-7B).
        projection_dim: hidden dim of the MLP projections (kept = d_model in
            the original GUI-Actor implementation).
        num_attention_heads: heads of the self-attention block over visual
            features.
        dropout_rate: dropout in self-attention and post-residual.
        null_init_std: std-dev for the truncated-normal init of the null patch.
    \"\"\"

    def __init__(
        self,
        d_model: int,
        projection_dim: int,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.1,
        null_init_std: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        self.projection_dim = projection_dim

        # ── encoder/decoder projections (identical to upstream) ──────────────
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )

        # ── self-attention over visual features (identical to upstream) ─────
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # ── NEW: learnable null-patch embedding ──────────────────────────────
        # Lives in the *encoder* representation space (before projection_enc),
        # mirroring how visual patch hidden states enter the head.
        self.null_patch = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.null_patch, mean=0.0, std=null_init_std)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _build_extended_target(
        labels: torch.Tensor,            # (n_dec, n_enc) binary mask of patches in bbox
        refusal_labels: torch.Tensor,    # (n_dec,)   bool / 0-1 mask
    ) -> torch.Tensor:
        \"\"\"
        Build target probability distribution over `n_enc + 1` items.
          * positive samples → renormalised mask, zero on null index
          * refusal samples  → one-hot at the null index (last)
        \"\"\"
        n_dec, n_enc = labels.shape
        device = labels.device
        target = torch.zeros(n_dec, n_enc + 1, device=device, dtype=torch.float32)

        refusal_mask = refusal_labels.bool()
        non_refusal_mask = ~refusal_mask

        if non_refusal_mask.any():
            lf = labels[non_refusal_mask].float()
            denom = lf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            target[non_refusal_mask, :n_enc] = lf / denom

        if refusal_mask.any():
            target[refusal_mask, n_enc] = 1.0

        return target

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        hidden_state_enc: torch.Tensor,            # (n_enc, d_model)
        hidden_state_dec: torch.Tensor,            # (n_dec, d_model)
        labels: Optional[torch.Tensor] = None,     # (n_dec, n_enc) binary mask
        refusal_labels: Optional[torch.Tensor] = None,  # (n_dec,)
        do_single_patch: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        \"\"\"
        Returns:
            attn_weights: (n_dec, n_enc + 1)
                Probabilities. Index `n_enc` (last) is the null/refusal slot.
            loss: scalar tensor or None.
        \"\"\"
        # ── self-attention contextualisation of visual features ─────────────
        enc_input = hidden_state_enc.unsqueeze(0)   # (1, n_enc, d_model)
        attn_output, _ = self.self_attention(
            query=enc_input, key=enc_input, value=enc_input, need_weights=False,
        )
        hidden_state_enc_ctx = self.layer_norm(
            enc_input + self.dropout(attn_output)
        ).squeeze(0)                                # (n_enc, d_model)

        # ── append the null patch on the encoder side ───────────────────────
        # `self.null_patch` is in raw d_model space, like visual hidden states.
        # We do NOT pass it through self-attention — the null patch is meant to
        # be context-free (it is the same regardless of what is on screen).
        enc_with_null = torch.cat(
            [hidden_state_enc_ctx, self.null_patch.to(hidden_state_enc_ctx.dtype)],
            dim=0,
        )                                            # (n_enc + 1, d_model)

        # ── projections ─────────────────────────────────────────────────────
        proj_enc = self.projection_enc(enc_with_null)        # (n_enc+1, d_model)
        proj_dec = self.projection_dec(hidden_state_dec)     # (n_dec,   d_model)

        # ── scaled dot-product attention ────────────────────────────────────
        scaling = self.d_model ** 0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling
        # patch_logits: (n_dec, n_enc + 1)
        attn_weights = F.softmax(patch_logits, dim=-1)

        # ── loss ────────────────────────────────────────────────────────────
        loss = None
        if labels is not None and not do_single_patch:
            n_dec, n_enc = labels.shape
            if refusal_labels is None:
                refusal_labels = torch.zeros(n_dec, device=labels.device, dtype=torch.bool)

            target_dist = self._build_extended_target(labels, refusal_labels)
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)
            loss = F.kl_div(pred_log_probs, target_dist, reduction=\"batchmean\")

        return attn_weights, loss

    # --------------------------------------------------------- weight transfer
    @torch.no_grad()
    def load_from_original(self, original_head: nn.Module) -> None:
        \"\"\"
        Copy weights from a vanilla `VisionHead_MultiPatch` so we can warm-start
        the action head before fine-tuning. Only the new `null_patch` parameter
        remains randomly initialised.
        \"\"\"
        self.projection_enc.load_state_dict(original_head.projection_enc.state_dict())
        self.projection_dec.load_state_dict(original_head.projection_dec.state_dict())
        self.self_attention.load_state_dict(original_head.self_attention.state_dict())
        self.layer_norm.load_state_dict(original_head.layer_norm.state_dict())
"
