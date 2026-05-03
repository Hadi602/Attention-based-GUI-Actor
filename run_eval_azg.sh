#!/bin/bash
# ============================================================
# Adaptive Zoom Grounding — Evaluation Script
# ============================================================
# Run this from the GUI-Actor root directory:
#   bash src/azg/run_eval_azg.sh
# ============================================================

# Configuration
MODEL_PATH="/data4/rashid_GUI/models/GUI-Actor-7B"
BENCHMARK="screenspot_pro"
ENTROPY_THRESHOLD=0.75
OUTPUT_DIR="/data4/rashid_GUI/outputs/eval_results_azg"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "========================================"
echo " Adaptive Zoom Grounding Evaluation"
echo "========================================"
echo " Model: $MODEL_PATH"
echo " Benchmark: $BENCHMARK"
echo " Entropy Threshold: $ENTROPY_THRESHOLD"
echo " Output: $OUTPUT_DIR"
echo "========================================"

# Run evaluation with AZG
python -u eval/screenSpot_pro_azg.py \
    --model_path $MODEL_PATH \
    --entropy_threshold $ENTROPY_THRESHOLD \
    --output_dir $OUTPUT_DIR \
    --verbose

echo "Evaluation complete. Results saved to $OUTPUT_DIR"
