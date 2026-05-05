"\"\"\"
Integration helper: monkey-patches GUI-Actor's `VisionHead_MultiPatch`
with `NullPatchAdapter` so you can warm-start from the published
GUI-Actor checkpoint without forking the upstream codebase.

Usage
-----
from src.scripts.integrate_with_gui_actor import patch_gui_actor

model = build_gui_actor(...)              # your existing builder
model = patch_gui_actor(model,
                        hidden_dim=3584,
                        patch_dim=3584,
                        tau_null=1.0)
# Now `model.action_head` returns dicts that include `p_refuse`.

If you've already cloned GUI-Actor under /data4/rashid_GUI:
    cd /data4/rashid_GUI && python -m src.scripts.integrate_with_gui_actor \
        --gui_actor_root /data4/rashid_GUI/GUI-Actor

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch.nn as nn

from src.models.action_head_with_null import (
    NullPatchAdapter, VisionHeadWithNull,
)


def patch_gui_actor(
    model: nn.Module,
    hidden_dim: int,
    patch_dim: Optional[int] = None,
    tau: float = 1.0,
    tau_null: float = 1.0,
    init_null_scale: float = 1e-3,
    head_attr_name: str = \"action_head\",
) -> nn.Module:
    \"\"\"
    Replace the named action-head attribute with a NullPatchAdapter
    that wraps the original. Original weights are preserved.

    Parameters
    ----------
    model : the GUI-Actor model (instance of GUIActorForConditionalGeneration
            or whatever class your fork uses).
    hidden_dim, patch_dim : dims for the new null-patch projection.
    head_attr_name : default \"action_head\" — change if your fork
            names it differently (e.g., \"vision_head\").
    \"\"\"
    if not hasattr(model, head_attr_name):
        raise AttributeError(
            f\"model has no attribute '{head_attr_name}'. Available \"
            f\"attributes containing 'head': \"
            f\"{[n for n in dir(model) if 'head' in n.lower()]}\"
        )

    original_head = getattr(model, head_attr_name)
    adapter = NullPatchAdapter(
        original_head=original_head,
        patch_dim=patch_dim or hidden_dim,
        hidden_dim=hidden_dim,
        tau=tau,
        tau_null=tau_null,
        init_null_scale=init_null_scale,
    )
    setattr(model, head_attr_name, adapter)
    print(f\"[patch_gui_actor] Replaced {head_attr_name} -> NullPatchAdapter \"
          f\"(adds {adapter.null_patch.numel() + adapter.null_bias.numel()} \"
          f\"new params, plus {sum(p.numel() for p in adapter.q_proj_null.parameters())} \"
          f\"in q_proj_null)\")
    return model


def replace_with_full_head(
    model: nn.Module,
    hidden_dim: int,
    patch_dim: Optional[int] = None,
    tau: float = 1.0,
    tau_null: float = 1.0,
    head_attr_name: str = \"action_head\",
    copy_weights: bool = True,
) -> nn.Module:
    \"\"\"
    Fully replace the head with a fresh `VisionHeadWithNull`. Optionally
    copies q_proj/k_proj weights from the original head when names match.
    \"\"\"
    original = getattr(model, head_attr_name)
    new = VisionHeadWithNull(
        hidden_dim=hidden_dim, patch_dim=patch_dim or hidden_dim,
        tau=tau, tau_null=tau_null,
    )
    if copy_weights:
        for src_name, src_param in original.named_parameters():
            if src_name in dict(new.named_parameters()):
                tgt = dict(new.named_parameters())[src_name]
                if tgt.shape == src_param.shape:
                    tgt.data.copy_(src_param.data)
                    print(f\"[replace_with_full_head] copied {src_name}\")
    setattr(model, head_attr_name, new)
    return model


# Optional CLI smoke-test
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--gui_actor_root\", required=True,
                    help=\"Path to your GUI-Actor clone (must be importable)\")
    ap.add_argument(\"--head_attr\", default=\"action_head\")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.gui_actor_root) / \"src\"))
    try:
        from gui_actor.modeling import VisionHead_MultiPatch  # noqa: F401
        print(\"[cli] imported VisionHead_MultiPatch OK\")
    except Exception as e:                                    # noqa: BLE001
        print(\"[cli] import failed:\", e)
        sys.exit(1)


if __name__ == \"__main__\":
    _cli()
"
