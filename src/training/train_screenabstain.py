"\"\"\"
ScreenAbstain training loop.

Wraps a (frozen-or-LoRA) GUI-Actor backbone whose action head has
been swapped for `VisionHeadWithNull` (or a `NullPatchAdapter`).
Trains on a mixed clean + refusal dataset with `ScreenAbstainLoss`.

Usage
-----
$ python -m src.training.train_screenabstain \
    --config experiments/configs/screenabstain_full.yaml

Design choices
--------------
- Default freezes the LM and vision encoder; only the action head
  + null_patch are trained. This is what makes the experiment
  cheap (~1 GPU-day on a single A6000).
- Optional LoRA on attention layers of the action head's q_proj /
  k_proj for slightly higher capacity (off by default).
- Mixed precision via torch.cuda.amp (bf16 if available).

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from src.models.action_head_with_null import VisionHeadWithNull
from src.training.loss import ScreenAbstainLoss, ScreenAbstainLossConfig
from src.data.datasets import (
    ScreenAbstainTrainDataset, TrainDatasetConfig, make_collate_fn,
)


# ---------------------------------------------------------------------- #
# Backbone wrapper — extracted so it can be swapped per checkpoint type.
# ---------------------------------------------------------------------- #
class GUIActorWithNull(torch.nn.Module):
    \"\"\"
    Minimal wrapper:
        backbone : returns (action_token_hidden, patch_embeddings, patch_mask)
        head     : VisionHeadWithNull
    \"\"\"

    def __init__(self, backbone: torch.nn.Module, head: VisionHeadWithNull):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch: Dict) -> Dict:
        # Backbone is expected to produce these three tensors. Adapt
        # this method to your specific GUI-Actor checkpoint.
        outputs = self.backbone(**{
            k: v for k, v in batch.items()
            if k not in (\"target_idx\", \"is_refusal\")
        })
        head_out = self.head(
            query=outputs[\"action_query\"],          # (B, D)
            patches=outputs[\"patch_embeddings\"],    # (B, N, D)
            patch_mask=outputs.get(\"patch_mask\"),
        )
        return head_out


# ---------------------------------------------------------------------- #
# Param freeze / unfreeze
# ---------------------------------------------------------------------- #
def configure_params(model: GUIActorWithNull, train_full_head: bool = True):
    \"\"\"Freeze backbone; train head (incl. null_patch) by default.\"\"\"
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = train_full_head
    # Always train null_patch + null_bias even if head is otherwise frozen.
    model.head.null_patch.requires_grad = True
    model.head.null_bias.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f\"[train] trainable params: {n_trainable:,} / {n_total:,} \"
          f\"({100*n_trainable/max(1,n_total):.4f}%)\")


# ---------------------------------------------------------------------- #
# Training step
# ---------------------------------------------------------------------- #
def train_one_epoch(
    model: GUIActorWithNull,
    loss_fn: ScreenAbstainLoss,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    epoch: int,
    log_every: int = 50,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    running = {\"loss\": 0.0, \"loss_ground\": 0.0, \"loss_refuse\": 0.0}
    n_steps = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp,
                                     dtype=torch.bfloat16):
            out = model(batch)
            loss_out = loss_fn(
                attn_logits=out[\"attn_logits\"],
                target_idx=batch[\"target_idx\"],
                is_refusal=batch[\"is_refusal\"],
            )
            loss = loss_out[\"loss\"]

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        scaler.step(opt)
        scaler.update()

        for k in running:
            running[k] += float(loss_out[k]) if k in loss_out else float(loss)
        n_steps += 1

        if (step + 1) % log_every == 0:
            avg = {k: v / n_steps for k, v in running.items()}
            print(f\"[ep {epoch} step {step+1}/{len(loader)}] \"
                  f\"loss={avg['loss']:.4f} \"
                  f\"L_g={avg['loss_ground']:.4f} \"
                  f\"L_r={avg['loss_refuse']:.4f}\")

    print(f\"[ep {epoch}] done in {time.time()-t0:.1f}s, \"
          f\"steps={n_steps}\")
    return {k: v / max(1, n_steps) for k, v in running.items()}


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def load_yaml(p: str) -> Dict:
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--config\", required=True)
    ap.add_argument(\"--output_dir\", default=\"experiments/results/run\")
    ap.add_argument(\"--device\", default=\"cuda\")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    ds_cfg = TrainDatasetConfig(**cfg[\"dataset\"])
    train_ds = ScreenAbstainTrainDataset(ds_cfg)

    # User must inject a `processor` callable matching their GUI-Actor
    # backbone (Qwen2-VL processor or similar). For scaffolding, we
    # accept a dotted path under cfg[\"processor_path\"].
    processor = _import_callable(cfg[\"processor_path\"])
    collate = make_collate_fn(processor)

    loader = DataLoader(
        train_ds, batch_size=cfg[\"train\"][\"batch_size\"],
        shuffle=True, num_workers=cfg[\"train\"].get(\"num_workers\", 4),
        collate_fn=collate, pin_memory=True,
    )

    # ---- Model ----
    backbone = _import_callable(cfg[\"backbone_path\"])()
    head = VisionHeadWithNull(**cfg[\"head\"])
    model = GUIActorWithNull(backbone, head).to(args.device)
    configure_params(model, train_full_head=cfg[\"train\"].get(\"train_full_head\", True))

    # ---- Optimizer / loss ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg[\"train\"][\"lr\"],
        weight_decay=cfg[\"train\"].get(\"weight_decay\", 0.01),
    )
    loss_fn = ScreenAbstainLoss(ScreenAbstainLossConfig(**cfg[\"loss\"]))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg[\"train\"].get(\"amp\", True))

    # ---- Train ----
    history = []
    for epoch in range(cfg[\"train\"][\"epochs\"]):
        stats = train_one_epoch(
            model, loss_fn, loader, opt, scaler,
            device=args.device, epoch=epoch,
            log_every=cfg[\"train\"].get(\"log_every\", 50),
            grad_clip=cfg[\"train\"].get(\"grad_clip\", 1.0),
            use_amp=cfg[\"train\"].get(\"amp\", True),
        )
        history.append(stats)
        ckpt = out / f\"ckpt_epoch{epoch}.pt\"
        torch.save({
            \"head_state\": model.head.state_dict(),
            \"config\": cfg,
            \"epoch\": epoch,
            \"stats\": stats,
        }, ckpt)
        print(f\"[ep {epoch}] saved {ckpt}\")

    with open(out / \"history.json\", \"w\") as f:
        json.dump(history, f, indent=2)


def _import_callable(dotted: str):
    \"\"\"Import 'pkg.mod:obj' or 'pkg.mod.obj' to a callable.\"\"\"
    if \":\" in dotted:
        mod_name, obj_name = dotted.split(\":\", 1)
    else:
        mod_name, obj_name = dotted.rsplit(\".\", 1)
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, obj_name)


if __name__ == \"__main__\":
    main()
"
