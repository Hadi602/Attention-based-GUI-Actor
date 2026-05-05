"\"\"\"
Dataset wrappers for ScreenAbstain training & eval.

Two flavours:

1. ScreenAbstainTrainDataset
   Mixes clean (groundable) samples + refusal samples produced by
   RefusalDataGenerator. Maintains a configurable refusal fraction.

2. VenusBenchGDDataset
   Reads VenusBench-GD format (released Dec 2025). Three splits:
       - groundable     (target visible, instruction matches)
       - refusal_target (target absent / cropped out)
       - refusal_instr  (instruction unsatisfiable on this screen)

3. ScreenSpotDataset
   Standard ScreenSpot regression eval — used to verify ScreenAbstain
   does NOT hurt clean grounding accuracy.

Note: collate_fn produces tensors that match the contract expected by
`VisionHeadWithNull` and `ScreenAbstainLoss`.

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def bbox_to_patch_idx(bbox: Tuple[int, int, int, int],
                      img_size: Tuple[int, int],
                      patch_grid: Tuple[int, int]) -> int:
    \"\"\"
    Map a bbox center to a flat patch index (row-major).
    GUI-Actor uses a fixed patch grid (e.g., 36×36) per image after
    its vision encoder downsamples. Adjust patch_grid to your model.
    \"\"\"
    W, H = img_size
    Gh, Gw = patch_grid
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    col = min(Gw - 1, max(0, int(cx / W * Gw)))
    row = min(Gh - 1, max(0, int(cy / H * Gh)))
    return row * Gw + col


# ---------------------------------------------------------------------- #
# Train-time mixed dataset
# ---------------------------------------------------------------------- #
@dataclass
class TrainDatasetConfig:
    clean_manifest: str               # jsonl
    refusal_manifest: str             # jsonl produced by materialize_to_disk
    refusal_fraction: float = 0.4     # share of refusal samples in epoch
    patch_grid: Tuple[int, int] = (36, 36)


class ScreenAbstainTrainDataset(Dataset):
    \"\"\"
    Mixes clean + refusal samples. Each __getitem__ returns the dict:
        {
            \"image\": PIL.Image,          # caller transforms to tensor
            \"instruction\": str,
            \"target_idx\": int,           # patch idx, or -1 for refusal
            \"is_refusal\": bool,
        }

    Image-to-tensor and tokenization are model-specific; pass a
    `processor` callable into your DataLoader's collate_fn.
    \"\"\"

    def __init__(self, cfg: TrainDatasetConfig) -> None:
        self.cfg = cfg
        self.clean = _load_jsonl(cfg.clean_manifest)
        self.refusal = _load_jsonl(cfg.refusal_manifest)

        # Compute lengths for desired refusal fraction.
        rf = cfg.refusal_fraction
        n_clean = len(self.clean)
        n_refuse_target = int(n_clean * rf / max(1e-6, 1 - rf))
        # Repeat / truncate refusal pool to match
        if len(self.refusal) == 0:
            self.refusal_idx = []
        else:
            reps = (n_refuse_target + len(self.refusal) - 1) // len(self.refusal)
            self.refusal_idx = (list(range(len(self.refusal))) * reps)[:n_refuse_target]

        self.length = n_clean + len(self.refusal_idx)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Dict:
        if i < len(self.clean):
            s = self.clean[i]
            img = Image.open(s[\"image_path\"]).convert(\"RGB\")
            target_idx = bbox_to_patch_idx(
                tuple(s[\"target_bbox\"]), img.size, self.cfg.patch_grid
            )
            return {
                \"image\": img,
                \"instruction\": s[\"instruction\"],
                \"target_idx\": target_idx,
                \"is_refusal\": False,
                \"source_id\": s.get(\"source_id\", \"\"),
            }
        else:
            j = self.refusal_idx[i - len(self.clean)]
            s = self.refusal[j]
            img = Image.open(s[\"image_path\"]).convert(\"RGB\")
            return {
                \"image\": img,
                \"instruction\": s[\"instruction\"],
                \"target_idx\": -1,
                \"is_refusal\": True,
                \"perturbation\": s.get(\"perturbation\", \"\"),
                \"source_id\": s.get(\"source_id\", \"\"),
            }


# ---------------------------------------------------------------------- #
# VenusBench-GD eval dataset
# ---------------------------------------------------------------------- #
class VenusBenchGDDataset(Dataset):
    \"\"\"
    Expects manifest jsonl with fields:
        image_path, instruction, split (groundable|refusal_target|refusal_instr),
        target_bbox (nullable)
    \"\"\"
    SPLIT_NAMES = (\"groundable\", \"refusal_target\", \"refusal_instr\")

    def __init__(self, manifest_path: str,
                 patch_grid: Tuple[int, int] = (36, 36),
                 split_filter: Optional[str] = None) -> None:
        items = _load_jsonl(manifest_path)
        if split_filter:
            assert split_filter in self.SPLIT_NAMES
            items = [x for x in items if x[\"split\"] == split_filter]
        self.items = items
        self.patch_grid = patch_grid

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        s = self.items[i]
        img = Image.open(s[\"image_path\"]).convert(\"RGB\")
        is_ref = s[\"split\"].startswith(\"refusal\")
        target_idx = -1
        if not is_ref and s.get(\"target_bbox\") is not None:
            target_idx = bbox_to_patch_idx(
                tuple(s[\"target_bbox\"]), img.size, self.patch_grid
            )
        return {
            \"image\": img,
            \"instruction\": s[\"instruction\"],
            \"target_idx\": target_idx,
            \"is_refusal\": is_ref,
            \"split\": s[\"split\"],
            \"source_id\": s.get(\"source_id\", \"\"),
            \"target_bbox\": s.get(\"target_bbox\"),
            \"image_size\": img.size,
        }


# ---------------------------------------------------------------------- #
# ScreenSpot regression eval
# ---------------------------------------------------------------------- #
class ScreenSpotDataset(Dataset):
    \"\"\"Minimal wrapper. Expects standard ScreenSpot jsonl manifest.\"\"\"

    def __init__(self, manifest_path: str,
                 patch_grid: Tuple[int, int] = (36, 36)) -> None:
        self.items = _load_jsonl(manifest_path)
        self.patch_grid = patch_grid

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        s = self.items[i]
        img = Image.open(s[\"image_path\"]).convert(\"RGB\")
        target_idx = bbox_to_patch_idx(
            tuple(s[\"target_bbox\"]), img.size, self.patch_grid
        )
        return {
            \"image\": img,
            \"instruction\": s[\"instruction\"],
            \"target_idx\": target_idx,
            \"is_refusal\": False,
            \"target_bbox\": s[\"target_bbox\"],
            \"image_size\": img.size,
            \"source_id\": s.get(\"source_id\", \"\"),
        }


# ---------------------------------------------------------------------- #
# Generic collate
# ---------------------------------------------------------------------- #
def make_collate_fn(processor: Callable):
    \"\"\"
    `processor` is a model-specific callable that takes a list of
    (image, instruction) and returns a dict of tensors compatible with
    the GUI-Actor backbone (e.g., Qwen2-VL processor).

    Returns batched dict with:
        pixel_values, input_ids, attention_mask, target_idx, is_refusal
    \"\"\"
    def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [b[\"image\"] for b in batch]
        instrs = [b[\"instruction\"] for b in batch]
        proc = processor(images, instrs)

        target_idx = torch.tensor([b[\"target_idx\"] for b in batch], dtype=torch.long)
        is_refusal = torch.tensor([b[\"is_refusal\"] for b in batch], dtype=torch.bool)

        proc[\"target_idx\"] = target_idx
        proc[\"is_refusal\"] = is_refusal
        return proc
    return _collate
"
