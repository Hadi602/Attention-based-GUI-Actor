"\"\"\"
Evaluate ScreenAbstain on VenusBench-GD.

Outputs
-------
- experiments/results/<run>/venusbench_predictions.jsonl
- experiments/results/<run>/venusbench_metrics.json
- experiments/results/<run>/venusbench_threshold_sweep.csv

Usage
-----
$ python -m src.eval.eval_venusbench \
    --config experiments/configs/eval_venusbench.yaml \
    --checkpoint experiments/results/run/ckpt_epoch9.pt \
    --output_dir experiments/results/run

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import VenusBenchGDDataset, make_collate_fn
from src.eval.metrics import compute_refusal_report, sweep_thresholds
from src.models.action_head_with_null import VisionHeadWithNull
from src.training.train_screenabstain import (
    GUIActorWithNull, _import_callable, load_yaml,
)


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    p_refuse_all, argmax_all = [], []
    is_ref_all, bboxes, img_sizes, splits, sids = [], [], [], [], []

    for batch in loader:
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        out = model(batch_dev)
        p_refuse_all.append(out[\"p_refuse\"].float().cpu().numpy())
        argmax_all.append(out[\"click_logits\"].argmax(-1).cpu().numpy())

        is_ref_all.extend(batch[\"is_refusal\"].tolist())
        bboxes.extend(batch.get(\"target_bbox\", [None] * len(batch[\"is_refusal\"])))
        img_sizes.extend(batch.get(\"image_size\", [(0, 0)] * len(batch[\"is_refusal\"])))
        splits.extend(batch.get(\"split\", [\"?\"] * len(batch[\"is_refusal\"])))
        sids.extend(batch.get(\"source_id\", [\"\"] * len(batch[\"is_refusal\"])))

    return {
        \"p_refuse\": np.concatenate(p_refuse_all),
        \"argmax_patch\": np.concatenate(argmax_all),
        \"is_refusal\": np.array(is_ref_all, dtype=bool),
        \"bboxes\": bboxes,
        \"img_sizes\": img_sizes,
        \"splits\": splits,
        \"source_ids\": sids,
    }


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

    # Build dataset / loader
    ds = VenusBenchGDDataset(cfg[\"manifest_path\"], patch_grid=patch_grid)
    processor = _import_callable(cfg[\"processor_path\"])
    loader = DataLoader(
        ds, batch_size=cfg.get(\"batch_size\", 8), shuffle=False,
        num_workers=cfg.get(\"num_workers\", 4),
        collate_fn=make_collate_fn(processor),
    )

    # Build model and load head
    backbone = _import_callable(cfg[\"backbone_path\"])()
    head = VisionHeadWithNull(**cfg[\"head\"])
    model = GUIActorWithNull(backbone, head).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.head.load_state_dict(ckpt[\"head_state\"])

    res = run_inference(model, loader, args.device)

    # Per-split metrics
    splits = np.array(res[\"splits\"])
    overall = compute_refusal_report(
        res[\"p_refuse\"], res[\"argmax_patch\"], res[\"is_refusal\"],
        res[\"bboxes\"], res[\"img_sizes\"], patch_grid=patch_grid,
        threshold=cfg.get(\"threshold\", 0.5),
    )

    per_split = {}
    for sname in (\"groundable\", \"refusal_target\", \"refusal_instr\"):
        m = splits == sname
        if m.sum() == 0:
            continue
        sub = compute_refusal_report(
            res[\"p_refuse\"][m], res[\"argmax_patch\"][m],
            res[\"is_refusal\"][m],
            [res[\"bboxes\"][i] for i in np.where(m)[0]],
            [res[\"img_sizes\"][i] for i in np.where(m)[0]],
            patch_grid=patch_grid,
            threshold=cfg.get(\"threshold\", 0.5),
        )
        per_split[sname] = sub.to_dict()

    sweep = sweep_thresholds(
        res[\"p_refuse\"], res[\"argmax_patch\"], res[\"is_refusal\"],
        res[\"bboxes\"], res[\"img_sizes\"], patch_grid=patch_grid,
    )

    # Dump
    with open(out / \"venusbench_metrics.json\", \"w\") as f:
        json.dump({\"overall\": overall.to_dict(),
                   \"per_split\": per_split}, f, indent=2)

    with open(out / \"venusbench_threshold_sweep.csv\", \"w\", newline=\"\") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader(); w.writerows(sweep)

    with open(out / \"venusbench_predictions.jsonl\", \"w\") as f:
        for i in range(len(res[\"p_refuse\"])):
            f.write(json.dumps({
                \"source_id\": res[\"source_ids\"][i],
                \"split\": res[\"splits\"][i],
                \"p_refuse\": float(res[\"p_refuse\"][i]),
                \"argmax_patch\": int(res[\"argmax_patch\"][i]),
                \"is_refusal_gt\": bool(res[\"is_refusal\"][i]),
            }) + \"
\")

    print(\"[eval] overall:\", overall.to_dict())
    print(\"[eval] wrote:\", out)


if __name__ == \"__main__\":
    main()
"
