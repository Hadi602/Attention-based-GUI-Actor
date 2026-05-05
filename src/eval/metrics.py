"\"\"\"
Refusal & grounding metrics for ScreenAbstain.

Definitions (all reported per split)
------------------------------------
On groundable split (target visible, instruction valid):
    GroundAcc       : % of samples where argmax patch lies inside gt bbox
                      AND p_refuse < threshold
    FalseRefuseRate : % of groundable samples flagged as refusal
                      (p_refuse ≥ threshold)
    AvgPRefuse_g    : mean p_refuse on groundable samples (lower=better)

On refusal split:
    TrueNegativeRate (TNR / RefuseAcc) : % flagged as refusal
    FalseClickRate   : % where model still picks a patch with high
                       confidence (p_refuse < threshold)
    AvgPRefuse_r     : mean p_refuse on refusal samples (higher=better)

Threshold-free:
    AUROC_refuse    : refusal vs groundable separation by p_refuse
    AUPRC_refuse

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
@dataclass
class RefusalReport:
    threshold: float
    n_groundable: int
    n_refusal: int

    ground_acc: float
    false_refuse_rate: float
    avg_p_refuse_g: float

    refuse_acc: float
    false_click_rate: float
    avg_p_refuse_r: float

    auroc_refuse: float
    auprc_refuse: float

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------- #
def _patch_in_bbox(patch_idx: int,
                   bbox: Tuple[int, int, int, int],
                   img_size: Tuple[int, int],
                   patch_grid: Tuple[int, int]) -> bool:
    Gh, Gw = patch_grid
    W, H = img_size
    row, col = divmod(patch_idx, Gw)
    px1 = col * (W / Gw)
    py1 = row * (H / Gh)
    px2 = px1 + (W / Gw)
    py2 = py1 + (H / Gh)
    cx, cy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _auc_curves(y_score: np.ndarray, y_true: np.ndarray
                ) -> Tuple[float, float]:
    \"\"\"
    Compute AUROC and AUPRC without sklearn dependency.
    y_true: 1 = refusal, 0 = groundable
    y_score: p_refuse
    \"\"\"
    if len(np.unique(y_true)) < 2:
        return float(\"nan\"), float(\"nan\")

    order = np.argsort(-y_score, kind=\"mergesort\")
    y_true = y_true[order]
    y_score = y_score[order]

    # AUROC via Mann-Whitney
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    ranks = np.empty_like(y_score)
    # average ranks for ties
    i = 0
    rank = 1.0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and y_score[j + 1] == y_score[i]:
            j += 1
        avg_rank = 0.5 * (rank + rank + (j - i))
        ranks[i:j + 1] = avg_rank
        rank += (j - i + 1)
        i = j + 1
    # ranks were assigned in DESC order; flip to ASC.
    ranks = (len(y_score) + 1) - ranks
    sum_pos_ranks = ranks[y_true == 1].sum()
    auroc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    # AUPRC via stepwise (precision over decreasing thresholds)
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(1, n_pos)
    # Trapezoidal
    auprc = float(np.trapz(precision, recall))
    return float(auroc), auprc


# ---------------------------------------------------------------------- #
def compute_refusal_report(
    p_refuse: np.ndarray,            # (M,)
    argmax_patch: np.ndarray,        # (M,)
    is_refusal: np.ndarray,          # (M,) bool
    bboxes: List[Optional[Tuple[int, int, int, int]]],
    img_sizes: List[Tuple[int, int]],
    patch_grid: Tuple[int, int] = (36, 36),
    threshold: float = 0.5,
) -> RefusalReport:
    \"\"\"All inputs aligned by index. `bboxes[i]` may be None for refusal.\"\"\"
    p_refuse = np.asarray(p_refuse, dtype=np.float64)
    argmax_patch = np.asarray(argmax_patch, dtype=np.int64)
    is_refusal = np.asarray(is_refusal, dtype=bool)

    g_mask = ~is_refusal
    r_mask = is_refusal
    n_g, n_r = int(g_mask.sum()), int(r_mask.sum())

    pred_refuse = p_refuse >= threshold

    # Groundable side
    if n_g > 0:
        in_box = np.array([
            _patch_in_bbox(int(argmax_patch[i]), tuple(bboxes[i]),
                           img_sizes[i], patch_grid)
            for i in range(len(p_refuse)) if g_mask[i]
        ], dtype=bool)
        not_refused = ~pred_refuse[g_mask]
        ground_acc = float((in_box & not_refused).mean())
        false_refuse_rate = float(pred_refuse[g_mask].mean())
        avg_p_g = float(p_refuse[g_mask].mean())
    else:
        ground_acc = false_refuse_rate = avg_p_g = float(\"nan\")

    # Refusal side
    if n_r > 0:
        refuse_acc = float(pred_refuse[r_mask].mean())
        false_click_rate = float((~pred_refuse[r_mask]).mean())
        avg_p_r = float(p_refuse[r_mask].mean())
    else:
        refuse_acc = false_click_rate = avg_p_r = float(\"nan\")

    # Threshold-free
    auroc, auprc = _auc_curves(p_refuse, is_refusal.astype(np.int64))

    return RefusalReport(
        threshold=threshold,
        n_groundable=n_g, n_refusal=n_r,
        ground_acc=ground_acc,
        false_refuse_rate=false_refuse_rate,
        avg_p_refuse_g=avg_p_g,
        refuse_acc=refuse_acc,
        false_click_rate=false_click_rate,
        avg_p_refuse_r=avg_p_r,
        auroc_refuse=auroc,
        auprc_refuse=auprc,
    )


def sweep_thresholds(
    p_refuse: np.ndarray, argmax_patch: np.ndarray,
    is_refusal: np.ndarray, bboxes, img_sizes,
    patch_grid=(36, 36),
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> List[Dict]:
    rows = []
    for t in thresholds:
        r = compute_refusal_report(
            p_refuse, argmax_patch, is_refusal, bboxes, img_sizes,
            patch_grid=patch_grid, threshold=t,
        )
        rows.append(r.to_dict())
    return rows
"
