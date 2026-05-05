"#!/usr/bin/env bash
# Run training. Edit GUI_ACTOR_ROOT to point at your local clone.
set -euo pipefail

export GUI_ACTOR_ROOT=\"${GUI_ACTOR_ROOT:-/data4/rashid_GUI/GUI-Actor}\"
export GUI_ACTOR_CHECKPOINT=\"${GUI_ACTOR_CHECKPOINT:-microsoft/GUI-Actor-7B-Qwen2-VL}\"

CONFIG=\"${1:-experiments/configs/screenabstain_full.yaml}\"
OUT=\"${2:-experiments/results/run_$(date +%Y%m%d_%H%M%S)}\"

python -m src.training.train_screenabstain \
    --config \"${CONFIG}\" \
    --output_dir \"${OUT}\"
"
