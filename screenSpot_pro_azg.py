"""
ScreenSpot-Pro Evaluation with Attention based Zoom Grounding
=======================================================
Modified evaluation script that uses AZG inference.

Usage:
    python eval/screenSpot_pro_azg.py \
        --model_path /data4/rashid_GUI/models/GUI-Actor-7B \
        --entropy_threshold 0.75 \
        --output_dir /data4/rashid_GUI/outputs/eval_results_azg
"""

import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN
from gui_actor.inference import ForceFollowTokensLogitsProcessor

# Import AZG
from azg.inference_azg import inference_with_adaptive_zoom


def load_model(model_path):
    """Load GUI-Actor model and processor."""
    print(f"Loading model from {model_path}...")
    
    model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, tokenizer, processor


def build_conversation(image_path, instruction):
    """Build conversation in GUI-Actor format."""
    image = Image.open(image_path).convert("RGB")
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a GUI grounding assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    return conversation


def check_hit(pred_point, bbox, threshold=0.05):
    """Check if predicted point is within the bounding box."""
    if pred_point is None:
        return False
    
    x, y = pred_point
    x1, y1, x2, y2 = bbox
    
    return x1 <= x <= x2 and y1 <= y <= y2


def main(args):
    # Load model
    model, tokenizer, processor = load_model(args.model_path)
    
    # Load annotations
    anno_path = "/data4/rashid_GUI/data/benchmarks/screenspot_pro/annotations/all.json"
    with open(anno_path) as f:
        annotations = json.load(f)
    
    print(f"Loaded {len(annotations)} samples")
    print(f"Entropy threshold: {args.entropy_threshold}")
    
    # Setup logits processor
    logits_processor = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]]
    )
    
    # Run evaluation
    results = []
    hits = 0
    zoom_triggered = 0
    
    for i, sample in enumerate(tqdm(annotations, desc="Evaluating")):
        image_path = os.path.join(
            "/data4/rashid_GUI/data/benchmarks/screenspot_pro/images",
            sample["file_name"]
        )
        
        if not os.path.exists(image_path):
            continue
        
        conversation = build_conversation(image_path, sample["instruction"])
        
        # Run AZG inference
        pred = inference_with_adaptive_zoom(
            conversation, model, tokenizer, processor,
            logits_processor=logits_processor,
            use_placeholder=True,
            entropy_threshold=args.entropy_threshold,
            zoom_padding=0.15,
            zoom_top_k_patches=9,
            min_zoom_region=0.2,
            verbose=args.verbose and i < 5  # verbose only for first 5
        )
        
        # Check hit
        pred_point = pred["topk_points"][0] if pred["topk_points"] else None
        bbox = sample["bbox_x1y1x2y2"]
        hit = check_hit(pred_point, bbox)
        
        if hit:
            hits += 1
        if pred["azg_triggered"]:
            zoom_triggered += 1
        
        results.append({
            "file_name": sample["file_name"],
            "instruction": sample["instruction"],
            "domain": sample.get("domain", "unknown"),
            "data_type": sample.get("data_type", "unknown"),
            "bbox": bbox,
            "pred_point": pred_point,
            "hit": hit,
            "azg_triggered": pred["azg_triggered"],
            "azg_entropy": pred["azg_normalized_entropy"],
            "azg_zoom_region": pred["azg_zoom_region"]
        })
        
        if (i + 1) % 100 == 0:
            acc = hits / (i + 1) * 100
            zoom_rate = zoom_triggered / (i + 1) * 100
            print(f"  [{i+1}/{len(annotations)}] Accuracy: {acc:.2f}% | Zoom rate: {zoom_rate:.1f}%")
    
    # Compute final metrics
    total = len(results)
    accuracy = hits / total * 100 if total > 0 else 0
    zoom_rate = zoom_triggered / total * 100 if total > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"RESULTS (AZG, threshold={args.entropy_threshold})")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy:.2f}% ({hits}/{total})")
    print(f"Zoom Triggered: {zoom_rate:.1f}% ({zoom_triggered}/{total})")
    
    # Per-domain breakdown
    from collections import defaultdict
    domain_stats = defaultdict(lambda: {"total": 0, "hits": 0, "zooms": 0})
    for r in results:
        key = f"{r['domain']}-{r['data_type']}"
        domain_stats[key]["total"] += 1
        domain_stats[key]["hits"] += int(r["hit"])
        domain_stats[key]["zooms"] += int(r["azg_triggered"])
    
    print(f"\n{'Category':<25} {'Acc%':>8} {'Zoom%':>8} {'Total':>6}")
    print("-" * 50)
    for key in sorted(domain_stats.keys()):
        s = domain_stats[key]
        acc = s["hits"] / s["total"] * 100
        zr = s["zooms"] / s["total"] * 100
        print(f"{key:<25} {acc:>7.2f}% {zr:>7.1f}% {s['total']:>6}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "azg_predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    metrics = {
        "entropy_threshold": args.entropy_threshold,
        "overall_accuracy": accuracy,
        "zoom_rate": zoom_rate,
        "total_samples": total,
        "hits": hits,
        "domain_breakdown": dict(domain_stats)
    }
    with open(os.path.join(args.output_dir, "azg_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--entropy_threshold", type=float, default=0.75)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
