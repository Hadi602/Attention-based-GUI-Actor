"#!/usr/bin/env bash
# Evaluate on VenusBench-GD and ScreenSpot.
set -euo pipefail

export GUI_ACTOR_ROOT=\"${GUI_ACTOR_ROOT:-/data4/rashid_GUI/GUI-Actor}\"
export GUI_ACTOR_CHECKPOINT=\"${GUI_ACTOR_CHECKPOINT:-microsoft/GUI-Actor-7B-Qwen2-VL}\"

CKPT=\"${1:?Usage: run_eval.sh <ckpt.pt> <run_dir>}\"
RUN=\"${2:?Usage: run_eval.sh <ckpt.pt> <run_dir>}\"

echo \"=== VenusBench-GD ===\"
python -m src.eval.eval_venusbench \
    --config experiments/configs/eval_venusbench.yaml \
    --checkpoint \"${CKPT}\" \
    --output_dir \"${RUN}\"

echo \"=== ScreenSpot regression ===\"
python -m src.eval.eval_screenspot \
    --config experiments/configs/eval_screenspot.yaml \
    --checkpoint \"${CKPT}\" \
    --output_dir \"${RUN}\"

echo \"Done. Reports in ${RUN}/\"
"
