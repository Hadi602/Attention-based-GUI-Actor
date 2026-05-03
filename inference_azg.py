"""
Attention-based Zoom Grounding — Full Inference Pipeline
===================================================
Modified GUI-Actor inference that automatically zooms into
uncertain regions for improved small-icon grounding.

This file wraps the original inference.py without modifying it.
It can be used as a drop-in replacement during evaluation.

Author: Rashid (Attention-based Zoom Grounding Project)

Usage:
    from azg.inference_azg import inference_with_adaptive_zoom
    
    pred = inference_with_adaptive_zoom(
        conversation, model, tokenizer, data_processor,
        entropy_threshold=0.75,
        zoom_padding=0.15,
        max_zoom_iterations=1
    )
"""

import torch
import copy
from PIL import Image
from typing import Optional, Dict, Any

# Import from original GUI-Actor
from gui_actor.inference import (
    inference,
    get_prediction_region_point,
    ForceFollowTokensLogitsProcessor
)
from gui_actor.constants import (
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN
)

# Import AZG modules
from azg.entropy import (
    compute_normalized_entropy,
    should_zoom,
    get_top_k_patch_indices
)
from azg.zoom import (
    compute_zoom_region,
    crop_and_resize,
    remap_coordinates,
    compute_effective_patch_size,
    ZoomRegion
)


def inference_with_adaptive_zoom(
    conversation: list,
    model,
    tokenizer,
    data_processor,
    logits_processor=None,
    use_placeholder: bool = False,
    topk: int = 5,
    # AZG parameters
    entropy_threshold: float = 0.75,
    zoom_padding: float = 0.15,
    zoom_top_k_patches: int = 9,
    min_zoom_region: float = 0.2,
    max_zoom_iterations: int = 1,
    always_zoom: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    GUI-Actor inference with Adaptive Zoom Grounding.
    
    Pipeline:
    1. Run standard inference → get attention map
    2. Compute entropy → if above threshold, trigger zoom
    3. Crop top-k region → resize to full resolution
    4. Re-run inference on cropped image
    5. Remap coordinates back to original space
    
    Args:
        conversation: Standard GUI-Actor conversation format
        model: Qwen2VLForConditionalGenerationWithPointer model
        tokenizer: Corresponding tokenizer
        data_processor: Qwen2VL data processor
        logits_processor: Optional custom logits processor
        use_placeholder: Whether to use placeholder mode
        topk: Number of top regions to return
        
        # AZG-specific parameters:
        entropy_threshold: Normalized entropy threshold to trigger zoom (0-1)
        zoom_padding: Padding ratio around the zoom region
        zoom_top_k_patches: Number of top patches to define zoom region
        min_zoom_region: Minimum zoom region size (fraction of image)
        max_zoom_iterations: Maximum number of zoom iterations
        always_zoom: If True, always zoom regardless of entropy (for ablation)
        verbose: Print debug information
    
    Returns:
        pred: Dictionary with prediction results including:
            - All standard GUI-Actor outputs
            - azg_triggered: Whether zoom was triggered
            - azg_entropy: Entropy of first-pass attention
            - azg_zoom_region: The zoom region used (if triggered)
            - azg_effective_patch_size: Effective resolution after zoom
            - topk_points: Final predicted points (remapped if zoomed)
    """
    
    # ─── Stage 1: Standard Inference ──────────────────────────────────────
    
    pred = inference(
        conversation, model, tokenizer, data_processor,
        logits_processor=logits_processor,
        use_placeholder=use_placeholder,
        topk=topk
    )
    
    # Add AZG metadata
    pred["azg_triggered"] = False
    pred["azg_entropy"] = None
    pred["azg_normalized_entropy"] = None
    pred["azg_zoom_region"] = None
    pred["azg_effective_patch_size"] = None
    pred["azg_original_topk_points"] = pred["topk_points"]
    
    # If no attention scores (text-only output), return as-is
    if pred["attn_scores"] is None:
        return pred
    
    # ─── Stage 2: Entropy Check ───────────────────────────────────────────
    
    attn_tensor = torch.tensor(pred["attn_scores"])
    n_width = pred["n_width"]
    n_height = pred["n_height"]
    
    norm_entropy = compute_normalized_entropy(attn_tensor)
    pred["azg_normalized_entropy"] = norm_entropy
    
    if verbose:
        print(f"[AZG] Normalized entropy: {norm_entropy:.4f} (threshold: {entropy_threshold})")
    
    # Decide whether to zoom
    trigger_zoom = always_zoom or (norm_entropy >= entropy_threshold)
    
    if not trigger_zoom:
        if verbose:
            print(f"[AZG] Entropy below threshold. Using standard prediction.")
        return pred
    
    # ─── Stage 3: Compute Zoom Region ─────────────────────────────────────
    
    pred["azg_triggered"] = True
    
    if verbose:
        print(f"[AZG] Zoom triggered! Computing zoom region...")
    
    # Get top-k patches for zoom region computation
    top_coords, top_values = get_top_k_patch_indices(
        attn_tensor, n_width, n_height, k=zoom_top_k_patches
    )
    
    zoom_region = compute_zoom_region(
        top_coords, n_width, n_height,
        padding_ratio=zoom_padding,
        min_region_size=min_zoom_region
    )
    
    pred["azg_zoom_region"] = {
        "x1": zoom_region.x1, "y1": zoom_region.y1,
        "x2": zoom_region.x2, "y2": zoom_region.y2
    }
    
    if verbose:
        print(f"[AZG] Zoom region: {zoom_region}")
    
    # ─── Stage 4: Crop and Re-inference ───────────────────────────────────
    
    # Extract the image from the conversation
    image = _extract_image_from_conversation(conversation)
    if image is None:
        if verbose:
            print("[AZG] Could not extract image from conversation. Falling back.")
        return pred
    
    # Compute effective patch size after zoom
    eff_patch = compute_effective_patch_size(image.size, zoom_region)
    pred["azg_effective_patch_size"] = {"width": eff_patch[0], "height": eff_patch[1]}
    
    if verbose:
        print(f"[AZG] Effective patch size after zoom: {eff_patch[0]:.1f}x{eff_patch[1]:.1f}px")
    
    # Crop and resize
    zoomed_image = crop_and_resize(image, zoom_region)
    
    # Create new conversation with zoomed image
    zoom_conversation = _replace_image_in_conversation(conversation, zoomed_image)
    
    # Re-run inference on zoomed image
    zoom_pred = inference(
        zoom_conversation, model, tokenizer, data_processor,
        logits_processor=logits_processor,
        use_placeholder=use_placeholder,
        topk=topk
    )
    
    # ─── Stage 5: Remap Coordinates ───────────────────────────────────────
    
    if zoom_pred["topk_points"] is not None:
        remapped_points = []
        for point in zoom_pred["topk_points"]:
            # point is (x, y) in normalized crop coordinates
            orig_x, orig_y = remap_coordinates(point[0], point[1], zoom_region)
            remapped_points.append((orig_x, orig_y))
        
        pred["topk_points"] = remapped_points
        pred["topk_values"] = zoom_pred["topk_values"]
        
        if verbose:
            print(f"[AZG] Remapped points: {remapped_points[:3]}")
    
    return pred


def _extract_image_from_conversation(conversation: list) -> Optional[Image.Image]:
    """Extract the PIL Image from the conversation format."""
    for msg in conversation:
        if msg.get("role") == "user":
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    img = content.get("image")
                    if isinstance(img, Image.Image):
                        return img
                    elif isinstance(img, str):
                        return Image.open(img)
    return None


def _replace_image_in_conversation(conversation: list, new_image: Image.Image) -> list:
    """Create a copy of conversation with the image replaced."""
    new_conv = copy.deepcopy(conversation)
    for msg in new_conv:
        if msg.get("role") == "user":
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    content["image"] = new_image
                    return new_conv
    return new_conv


# ─── Evaluation Wrapper ───────────────────────────────────────────────────

def evaluate_with_azg(
    dataset,
    model,
    tokenizer,
    data_processor,
    entropy_threshold: float = 0.75,
    verbose: bool = False
) -> dict:
    """
    Run evaluation on a dataset with AZG enabled.
    
    Args:
        dataset: List of evaluation samples
        model: GUI-Actor model
        tokenizer: Tokenizer
        data_processor: Data processor
        entropy_threshold: Entropy threshold for zoom
        verbose: Print progress
    
    Returns:
        results: Dict with accuracy metrics and per-sample predictions
    """
    predictions = []
    zoom_triggered_count = 0
    
    for i, sample in enumerate(dataset):
        conversation = sample["conversation"]
        
        pred = inference_with_adaptive_zoom(
            conversation, model, tokenizer, data_processor,
            entropy_threshold=entropy_threshold,
            verbose=verbose
        )
        
        if pred["azg_triggered"]:
            zoom_triggered_count += 1
        
        predictions.append({
            "sample_id": i,
            "pred_point": pred["topk_points"][0] if pred["topk_points"] else None,
            "azg_triggered": pred["azg_triggered"],
            "azg_entropy": pred["azg_normalized_entropy"],
            "azg_zoom_region": pred["azg_zoom_region"],
            **sample.get("metadata", {})
        })
        
        if verbose and (i + 1) % 50 == 0:
            print(f"[AZG Eval] Processed {i+1}/{len(dataset)} | Zoom triggered: {zoom_triggered_count}")
    
    results = {
        "total_samples": len(dataset),
        "zoom_triggered": zoom_triggered_count,
        "zoom_rate": zoom_triggered_count / len(dataset) if dataset else 0,
        "predictions": predictions
    }
    
    return results
