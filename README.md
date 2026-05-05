"# ScreenAbstain — A Null-Patch Action Head for Refusal-Aware Coordinate-Free GUI Grounding

> A research extension of **GUI-Actor** (Microsoft Research, NeurIPS 2025) that endows the model with the ability to *abstain* when an instruction refers to a UI element that is not present on screen.

## TL;DR

Coordinate-free GUI grounding models such as GUI-Actor compute a softmax over *existing* image patches, so they cannot say \"no target on screen\" — they always click somewhere. On the brand-new **VenusBench-GD Refusal Grounding** benchmark (arXiv 2512.16501, Dec 2025), 14 of 16 SOTA GUI models score 0.00–0.22% accuracy.

We add a single learnable **null-patch** to the action head:

```
softmax over [patch_1, patch_2, ..., patch_N, null_patch]
                                              ^^^^^^^^^
                                              new — represents \"no target on screen\"
```

Training: keep VLM backbone and verifier frozen, only train the action head with mixed positive (standard grounding) + synthetic refusal data.

## Key results (target)

| Method                      | ScreenSpot-Pro | VenusBench-GD Refusal |
| --------------------------- | -------------: | --------------------: |
| GUI-Actor-7B (reproduced)   |          41.4% |                  0.0% |
| GUI-Actor-7B + entropy thresh (baseline) |  ~40% |               ~10–15% |
| **GUI-Actor-7B + null-patch (ours)**     |  ≥ 40% |              **≥ 30%** |

## Repo layout

```
Attention-based-GUI-Actor/
├── README.md                  ← you are here
├── GUIDE.md                   ← step-by-step daily plan
├── requirements.txt
├── setup.py
├── docs/
│   ├── 00_motivation.md
│   ├── 01_related_work.md
│   ├── 02_method.md           ← method specification
│   └── 03_experiments_plan.md
├── data/                      ← (gitignored) downloaded benchmarks + synthetic data
├── src/screen_abstain/
│   ├── models/
│   │   ├── action_head_with_null.py   ⭐ core contribution
│   │   └── patched_model.py            wrapper around GUI-Actor
│   ├── data/
│   │   ├── refusal_perturbation.py    synthetic refusal generator
│   │   ├── venusbench_loader.py
│   │   └── inspect_refusal_data.py
│   ├── training/
│   │   ├── train_action_head.py
│   │   └── losses.py
│   └── eval/
│       ├── evaluate_refusal.py
│       ├── evaluate_baseline_entropy.py
│       └── evaluate_screenspot_pro.py
├── scripts/                   bash entry points
└── experiments/
    ├── configs/
    └── results/
```

## Quick start

```bash
conda activate gui_actor
pip install -e .
bash scripts/00_setup.sh
bash scripts/01_run_baseline_on_venusbench.sh   # produces the 0% baseline number
```

For the full plan see [`GUIDE.md`](./GUIDE.md).

## Citation

```bibtex
@misc{screenabstain2026,
  title  = {ScreenAbstain: A Null-Patch Action Head for Refusal-Aware Coordinate-Free GUI Grounding},
  author = {Hadi and ...},
  year   = {2026},
  note   = {arXiv preprint (forthcoming)}
}
```

We build on GUI-Actor:
```bibtex
@inproceedings{wu2025guiactor,
  title  = {GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents},
  author = {Wu, Qianhui and others},
  booktitle = {NeurIPS},
  year   = {2025}
}
```

And evaluate on VenusBench-GD:
```bibtex
@misc{zhou2025venusbench,
  title  = {VenusBench-GD: A Comprehensive Multi-Platform GUI Benchmark for Diverse Grounding Tasks},
  author = {Zhou, Beitong and others},
  year   = {2025},
  eprint = {2512.16501},
  archivePrefix = {arXiv}
}
```
"
