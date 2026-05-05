"\"\"\"
ScreenAbstain: Refusal-aware action head for coordinate-free GUI grounding.

Drop-in replacement for GUI-Actor's `VisionHead_MultiPatch`
(src/gui_actor/modeling.py). Adds a single learnable `null_patch`
parameter that competes with real visual patches in the attention
softmax. When the model attends to `null_patch`, it abstains.

Key design decisions
--------------------
1. Architectural, not a training trick: `null_patch` lives inside the
   action head as a learned KEY vector that participates in the
   patch-attention softmax. This makes refusal a first-class action.
2. Coordinate-free: output remains an attention distribution over
   {N visual patches + 1 null_patch}. Final click coordinate is the
   argmax/expected position over the N visual patches *only after*
   verifying argmax != null index.
3. Single new parameter (D-dim vector) — minimal capacity addition,
   easy to ablate.
4. Optional temperature `tau_null` lets us calibrate refusal
   sensitivity post-hoc without retraining.

Reference
---------
Replaces: `VisionHead_MultiPatch` in GUI-Actor
  (Wu et al., \"GUI-Actor: Coordinate-Free Visual Grounding ...\", 2025)

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionHeadWithNull(nn.Module):
    \"\"\"
    Coordinate-free action head augmented with a learnable null patch.

    Forward pass produces:
        attn_logits : (B, N+1)  raw logits over [N visual patches | null]
        attn_probs  : (B, N+1)  softmax of attn_logits / tau
        p_refuse    : (B,)      probability mass on null index
        click_logits: (B, N)    logits over visual patches only
                                (use these for coordinate decoding)

    Args
    ----
    hidden_dim : int       Size of LM hidden states (D).
    patch_dim  : int       Size of visual patch embeddings (D_v).
                           In GUI-Actor this == hidden_dim after projection.
    tau        : float     Softmax temperature. Default 1.0.
    tau_null   : float     Extra scalar multiplier on the null logit
                           (post-hoc calibration knob). Default 1.0.
    init_null_scale : float Std of Gaussian init for null_patch. Small
                           init keeps refusal \"off\" early in training.
    use_proj   : bool      If True, linearly project query/key.
                           Matches GUI-Actor's MultiPatch head.
    \"\"\"

    def __init__(
        self,
        hidden_dim: int,
        patch_dim: Optional[int] = None,
        tau: float = 1.0,
        tau_null: float = 1.0,
        init_null_scale: float = 1e-3,
        use_proj: bool = True,
    ) -> None:
        super().__init__()

        patch_dim = patch_dim if patch_dim is not None else hidden_dim
        self.hidden_dim = hidden_dim
        self.patch_dim = patch_dim
        self.tau = tau
        self.tau_null = tau_null
        self.use_proj = use_proj

        if use_proj:
            self.q_proj = nn.Linear(hidden_dim, patch_dim, bias=False)
            self.k_proj = nn.Linear(patch_dim, patch_dim, bias=False)
        else:
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()

        # The single new parameter introduced by ScreenAbstain.
        # Shape (patch_dim,) — acts as an extra KEY in attention.
        self.null_patch = nn.Parameter(
            torch.randn(patch_dim) * init_null_scale
        )

        # Learnable bias on null logit (lets the model adjust refusal
        # threshold per-layer, optional but cheap).
        self.null_bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def num_new_params(self) -> int:
        \"\"\"Sanity check: how many parameters did we add?\"\"\"
        return self.null_patch.numel() + self.null_bias.numel()

    def get_null_index(self, num_patches: int) -> int:
        \"\"\"Convention: null is the LAST index.\"\"\"
        return num_patches

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        query: torch.Tensor,           # (B, D) action token hidden state
        patches: torch.Tensor,         # (B, N, D_v) visual patch embeddings
        patch_mask: Optional[torch.Tensor] = None,  # (B, N) 1=valid, 0=pad
    ) -> Dict[str, torch.Tensor]:
        \"\"\"
        Compute attention over [visual patches | null] and split outputs.
        \"\"\"
        B, N, _ = patches.shape

        q = self.q_proj(query)              # (B, D')
        k = self.k_proj(patches)            # (B, N, D')

        # Visual patch logits: scaled dot-product
        scale = q.shape[-1] ** -0.5
        vis_logits = torch.einsum(\"bd,bnd->bn\", q, k) * scale  # (B, N)

        if patch_mask is not None:
            vis_logits = vis_logits.masked_fill(~patch_mask.bool(), -1e4)

        # Null logit: same dot-product, broadcast null_patch per sample
        null_k = self.k_proj(self.null_patch).unsqueeze(0)        # (1, D')
        null_logit = (q * null_k).sum(dim=-1) * scale             # (B,)
        null_logit = null_logit * self.tau_null + self.null_bias  # (B,)

        # Concatenate: [vis | null]  →  (B, N+1)
        all_logits = torch.cat([vis_logits, null_logit.unsqueeze(-1)], dim=-1)
        all_probs = F.softmax(all_logits / self.tau, dim=-1)

        p_refuse = all_probs[:, -1]              # (B,)
        # Click logits: visual patches only (for coordinate decoding)
        click_logits = vis_logits                # un-normalized
        click_probs_given_act = F.softmax(vis_logits / self.tau, dim=-1)

        return {
            \"attn_logits\": all_logits,         # (B, N+1)
            \"attn_probs\": all_probs,           # (B, N+1)
            \"p_refuse\": p_refuse,              # (B,)
            \"click_logits\": click_logits,      # (B, N)
            \"click_probs_given_act\": click_probs_given_act,  # (B, N)
            \"null_index\": self.get_null_index(N),
        }

    # ------------------------------------------------------------------ #
    # Inference helper
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        query: torch.Tensor,
        patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        refuse_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        \"\"\"
        Return a discrete action: either refuse, or argmax patch index.

        Output keys:
            action     : (B,) int  — patch index, or -1 if refused
            p_refuse   : (B,) float
            refused    : (B,) bool
        \"\"\"
        out = self.forward(query, patches, patch_mask)
        p_refuse = out[\"p_refuse\"]
        refused = p_refuse >= refuse_threshold

        argmax_patch = out[\"click_logits\"].argmax(dim=-1)        # (B,)
        action = torch.where(refused, torch.full_like(argmax_patch, -1),
                             argmax_patch)
        return {
            \"action\": action,
            \"p_refuse\": p_refuse,
            \"refused\": refused,
            \"argmax_patch\": argmax_patch,
        }


# ====================================================================== #
# Adapter that wraps GUI-Actor's existing VisionHead_MultiPatch.
# Use this when you don't want to fork GUI-Actor — monkey-patch the
# action head at load time instead.
# ====================================================================== #
class NullPatchAdapter(nn.Module):
    \"\"\"
    Wraps an existing GUI-Actor `VisionHead_MultiPatch` and adds a
    learnable null patch on top, *without* modifying the original
    module's weights. Useful for warm-starting from a pretrained
    GUI-Actor checkpoint.

    Strategy: forward through the original head to get visual logits,
    then concat a null logit derived from a fresh learnable key.
    \"\"\"

    def __init__(
        self,
        original_head: nn.Module,
        patch_dim: int,
        hidden_dim: int,
        tau: float = 1.0,
        tau_null: float = 1.0,
        init_null_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.original_head = original_head
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.tau_null = tau_null

        # Re-use the original q_proj / k_proj if exposed; otherwise
        # learn fresh projections that map into patch space.
        self.q_proj_null = nn.Linear(hidden_dim, patch_dim, bias=False)
        self.null_patch = nn.Parameter(
            torch.randn(patch_dim) * init_null_scale
        )
        self.null_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        query: torch.Tensor,
        patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Run original head — expected to return (B, N) logits or dict.
        orig_out = self.original_head(query, patches, patch_mask, **kwargs)
        if isinstance(orig_out, dict):
            vis_logits = orig_out.get(\"logits\", orig_out.get(\"attn_logits\"))
        else:
            vis_logits = orig_out

        if vis_logits.dim() == 3:
            # GUI-Actor MultiPatch returns (B, N, num_heads); reduce.
            vis_logits = vis_logits.mean(dim=-1)

        if patch_mask is not None:
            vis_logits = vis_logits.masked_fill(~patch_mask.bool(), -1e4)

        scale = self.patch_dim ** -0.5
        q = self.q_proj_null(query)                        # (B, D')
        null_logit = (q * self.null_patch).sum(dim=-1) * scale
        null_logit = null_logit * self.tau_null + self.null_bias

        all_logits = torch.cat([vis_logits, null_logit.unsqueeze(-1)], dim=-1)
        all_probs = F.softmax(all_logits / self.tau, dim=-1)

        return {
            \"attn_logits\": all_logits,
            \"attn_probs\": all_probs,
            \"p_refuse\": all_probs[:, -1],
            \"click_logits\": vis_logits,
            \"click_probs_given_act\": F.softmax(vis_logits / self.tau, dim=-1),
            \"null_index\": vis_logits.shape[-1],
        }
"
