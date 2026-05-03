# Attention-based-GUI-Actor
GUI-Actor produces a native spatial attention map that encodes richer geometric uncertainty. We exploit this structure to build a zoom refinement module that is architecturally native to GUI-Actor, requires no retraining, and improves icon grounding accuracy on professional software domains.
# Attention-based Zoom Grounding (AZG)

> Improving GUI-Actor for professional software icon grounding via 
> entropy-gated inference-time adaptive zoom.

## Key Results

| Benchmark | Baseline | AZG (Ours) | Improvement |
|-----------|----------|------------|-------------|
| ScreenSpot-Pro (Overall) | 41.37% | TBD | TBD |
| CAD Icons | 9.38% | TBD | TBD |
| Creative Icons | 10.49% | TBD | TBD |
| Scientific Icons | 22.73% | TBD | TBD |

## Method

GUI-Actor uses a fixed 28×28px patch size in its vision encoder. 
Icons smaller than this (common in professional software) lose 
visual signal when merged with neighboring patches.

**AZG Solution:** When the action head's attention entropy is high 
(indicating uncertainty), we automatically crop the high-probability 
region and re-run inference at higher effective resolution.

```
Full Image → Action Head → Entropy HIGH?
                                ↓ YES
                         Crop top-k patches
                                ↓
                         Resize to full resolution
                                ↓
                         Re-run Action Head
                                ↓
                         Remap coordinates → Final prediction
```

**No retraining needed.** This is purely an inference-time modification.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-zoom-grounding.git
cd adaptive-zoom-grounding
pip install -e .
```

## Usage

```python
from azg.inference_azg import inference_with_adaptive_zoom

pred = inference_with_adaptive_zoom(
    conversation, model, tokenizer, processor,
    entropy_threshold=0.75,
    zoom_padding=0.15
)
```

## Evaluation

```bash
python eval/screenSpot_pro_azg.py \
    --model_path /path/to/GUI-Actor-7B \
    --entropy_threshold 0.75 \
    --output_dir outputs/azg_results
```

## Citation

```bibtex
@article{gui-actor-2025,
  title={GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents},
  author={...},
  journal={arXiv preprint arXiv:2506.03143},
  year={2025}
}
```

## AZG Implementation Code
Ready-to-use Python files. Copy these to your server at src/azg/

## How to use the code:
On server:
cd /data4/rashid_GUI/GUI-Actor
mkdir -p src/azg
touch src/azg/__init__.py

Put all files in the server into src/azg and then run the evaluation;
python eval/screenSpot_pro_azg.py \
  --model_path /data4/rashid_GUI/models/GUI-Actor-7B \
  --entropy_threshold 0.75 \
  --output_dir /data4/rashid_GUI/outputs/eval_results_azg




  ## Attention-based-zoom-grounding
```
  adaptive-zoom-grounding/
├── README.md                           # Project overview & results
├── LICENSE
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
│
├── src/
│   └── azg/                            # Core AZG package
│       ├── __init__.py
│       ├── entropy.py                  # Entropy computation module
│       ├── zoom.py                     # Crop & resize utilities
│       └── inference_azg.py            # Full AZG inference pipeline
│
├── eval/
│   ├── screenSpot_pro_azg.py           # ScreenSpot-Pro with AZG
│   ├── run_eval_azg.sh                 # Evaluation launch script
│   └── compare_results.py             # Baseline vs AZG comparison
│
├── experiments/
│   ├── configs/
│   │   ├── threshold_ablation.yaml     # Entropy threshold configs
│   │   └── zoom_radius_ablation.yaml   # Zoom region size configs
│   ├── results/                        # Experiment outputs (gitignored)
│   └── run_ablations.sh               # Run all ablation experiments
│
├── notebooks/
│   ├── 01_failure_analysis.ipynb       # Original failure analysis
│   ├── 02_entropy_visualization.ipynb  # Entropy distribution study
│   └── 03_results_comparison.ipynb     # Final results & figures
│
├── docs/
│   ├── method.md                       # Detailed method description
│   ├── architecture.md                 # GUI-Actor architecture notes
│   ├── reproduction.md                 # Reproduction notes
│   └── figures/                        # Paper figures
│
└── paper/
    ├── main.tex                        # Workshop paper (4 pages)
    ├── references.bib                  # Bibliography
    └── figures/                         # LaTeX figures
```
