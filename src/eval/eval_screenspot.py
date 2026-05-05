"\"\"\"
Regression eval on ScreenSpot.

Goal: confirm ScreenAbstain does NOT degrade clean grounding accuracy
relative to vanilla GUI-Actor. Reports:
    GroundAcc@click   = % of samples whose argmax-patch center lies
                        inside gt bbox (mirrors ScreenSpot click acc)
    Mean p_refuse     = should be near 0 on this split
    FalseRefuseRate   = % of clean samples flagged as refusal

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import ScreenSpotDataset, make_collate_fn
from src.eval.metrics import compute_refusal_report
from src.models.action_head_with_null import VisionHeadWithNull
from src.training.train_screenabstain import (
    GUIActorWithNull, _import_callable, load_yaml,
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--config\", required=True)
    ap.add_argument(\"--checkpoint\", required=True)
    ap.add_argument(\"--output_dir\", required=True)
    ap.add_argument(\"--device\", default=\"cuda\")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    patch_grid = tuple(cfg.get(\"patch_grid\", [36, 36]))

    ds = ScreenSpotDataset(cfg[\"manifest_path\"], patch_grid=patch_grid)
    processor = _import_callable(cfg[\"processor_path\"])
    loader = DataLoader(
        ds, batch_size=cfg.get(\"batch_size\", 8), shuffle=False,
        num_workers=cfg.get(\"num_workers\", 4),
        collate_fn=make_collate_fn(processor),
    )

    backbone = _import_callable(cfg[\"backbone_path\"])()
    head = VisionHeadWithNull(**cfg[\"head\"])
    model = GUIActorWithNull(backbone, head).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.head.load_state_dict(ckpt[\"head_state\"])
    model.eval()

    p_ref, argmax, bboxes, sizes = [], [], [], []
    for batch in loader:
        batch_dev = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        o = model(batch_dev)
        p_ref.append(o[\"p_refuse\"].float().cpu().numpy())
        argmax.append(o[\"click_logits\"].argmax(-1).cpu().numpy())
        bboxes.extend(batch[\"target_bbox\"])
        sizes.extend(batch[\"image_size\"])

    p_ref = np.concatenate(p_ref); argmax = np.concatenate(argmax)
    is_refusal = np.zeros(len(p_ref), dtype=bool)

    rep = compute_refusal_report(
        p_ref, argmax, is_refusal, bboxes, sizes,
        patch_grid=patch_grid,
        threshold=cfg.get(\"threshold\", 0.5),
    )

    payload = {
        \"ground_acc\": rep.ground_acc,
        \"false_refuse_rate\": rep.false_refuse_rate,
        \"avg_p_refuse\": rep.avg_p_refuse_g,
        \"n_samples\": int(len(p_ref)),
    }
    with open(out / \"screenspot_metrics.json\", \"w\") as f:
        json.dump(payload, f, indent=2)
    print(\"[screenspot]\", payload)


if __name__ == \"__main__\":
    main()
"
