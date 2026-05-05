"\"\"\"
Thin adapter that exposes a uniform interface to GUI-Actor's backbone.
You point this at your local GUI-Actor clone and it returns a model
whose forward(...) yields the keys our training loop expects:
    {\"action_query\": (B, D),
     \"patch_embeddings\": (B, N, D),
     \"patch_mask\": (B, N)}

This file is the ONLY one you should normally need to edit when
porting between GUI-Actor versions.

The default implementation imports `gui_actor.modeling` from the path
configured in env GUI_ACTOR_ROOT; if that path is empty or import
fails, we fall back to a tiny stub backbone for unit tests.

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------- #
# Real path: import GUI-Actor and wrap it.
# ---------------------------------------------------------------------- #
def _try_import_gui_actor():
    root = os.environ.get(\"GUI_ACTOR_ROOT\", \"\")
    if not root:
        return None
    src = str(Path(root) / \"src\")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from gui_actor.modeling import (                      # noqa: F401
            GUIActorForConditionalGeneration as GAModel,
        )
        from gui_actor.processing_qwen2_vl import (
            Qwen2VLProcessor as GAProcessor,
        )
        return GAModel, GAProcessor
    except Exception as e:                                    # noqa: BLE001
        print(f\"[backbone] GUI-Actor import failed: {e}; \"
              f\"falling back to stub.\")
        return None


# ---------------------------------------------------------------------- #
# Stub backbone for development without GUI-Actor available.
# ---------------------------------------------------------------------- #
class StubBackbone(nn.Module):
    \"\"\"
    Returns random tensors with the right shapes so the rest of the
    pipeline can be unit-tested. Replace this in production.
    \"\"\"

    def __init__(self, hidden_dim: int = 3584,
                 num_patches: int = 36 * 36) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        # one trainable param so the module is non-empty for autograd
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, **batch) -> Dict[str, torch.Tensor]:
        # Find batch size from any tensor in batch
        B = 1
        for v in batch.values():
            if torch.is_tensor(v):
                B = v.shape[0]
                break
        device = self._dummy.device
        return {
            \"action_query\": torch.randn(B, self.hidden_dim, device=device),
            \"patch_embeddings\": torch.randn(
                B, self.num_patches, self.hidden_dim, device=device
            ),
            \"patch_mask\": torch.ones(B, self.num_patches,
                                     dtype=torch.bool, device=device),
        }


class StubProcessor:
    def __call__(self, images: List, instructions: List[str]) -> Dict:
        # Returns minimal tensor dict; backbone is shape-agnostic.
        return {
            \"pixel_values\": torch.zeros(len(images), 3, 224, 224),
            \"input_ids\": torch.zeros(len(images), 16, dtype=torch.long),
            \"attention_mask\": torch.ones(len(images), 16, dtype=torch.long),
        }


# ---------------------------------------------------------------------- #
# Public builders (used by configs via dotted-path imports)
# ---------------------------------------------------------------------- #
def build_backbone() -> nn.Module:
    \"\"\"
    Returns a backbone whose forward(**batch) emits action_query,
    patch_embeddings, patch_mask.
    \"\"\"
    imp = _try_import_gui_actor()
    if imp is None:
        return StubBackbone()

    GAModel, _ = imp
    ckpt = os.environ.get(\"GUI_ACTOR_CHECKPOINT\")
    if ckpt:
        model = GAModel.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
    else:
        model = GAModel.from_pretrained(\"microsoft/GUI-Actor-7B-Qwen2-VL\")

    return _GUIActorWrapper(model)


def build_processor():
    imp = _try_import_gui_actor()
    if imp is None:
        return StubProcessor()
    _, GAProcessor = imp
    ckpt = os.environ.get(\"GUI_ACTOR_CHECKPOINT\",
                          \"microsoft/GUI-Actor-7B-Qwen2-VL\")
    proc = GAProcessor.from_pretrained(ckpt)

    def _call(images, instructions):
        return proc(text=instructions, images=images,
                    return_tensors=\"pt\", padding=True)
    return _call


class _GUIActorWrapper(nn.Module):
    \"\"\"
    Wraps the upstream GUI-Actor model so its forward yields the
    three tensors our training loop expects. You may need to tweak
    the field names depending on your exact checkpoint version.
    \"\"\"

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, **batch) -> Dict[str, torch.Tensor]:
        out = self.model(
            **{k: v for k, v in batch.items() if not k.startswith(\"_\")},
            return_dict=True, output_hidden_states=True,
        )
        # Field names below match the upstream `GUI-Actor` repo
        # commit referenced in /app/proposal. If your fork differs,
        # adjust here only.
        return {
            \"action_query\":  out.action_query,            # (B, D)
            \"patch_embeddings\": out.patch_embeddings,     # (B, N, D)
            \"patch_mask\":    getattr(out, \"patch_mask\", None),
        }
"
