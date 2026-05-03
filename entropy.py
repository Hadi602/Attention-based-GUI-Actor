"""
Attention-based Zoom Grounding — Entropy Module
=========================================
Computes Shannon entropy of attention distributions from GUI-Actor's
action head to determine if the model is uncertain and needs zoom.

Author: Rashid (Attention-based Zoom Grounding Project)
"""

import torch
import numpy as np
from typing import Tuple


def compute_attention_entropy(attn_scores: torch.Tensor) -> float:
    """
    Compute Shannon entropy of the attention distribution.
    
    High entropy = uniform distribution = model is uncertain
    Low entropy = peaked distribution = model is confident
    
    Args:
        attn_scores: Tensor of shape [1, n_patches] — softmax attention 
                     weights from VisionHead_MultiPatch
    
    Returns:
        entropy: float — Shannon entropy in bits
    """
    # Ensure we work with probabilities (already softmax from action head)
    probs = attn_scores[0]  # Remove batch dim: [n_patches]
    
    # Clamp to avoid log(0)
    probs = torch.clamp(probs, min=1e-10)
    
    # Shannon entropy: H = -sum(p * log2(p))
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    
    return entropy


def compute_max_entropy(n_patches: int) -> float:
    """
    Compute maximum possible entropy for n_patches.
    Max entropy = log2(n_patches), achieved when distribution is uniform.
    
    Args:
        n_patches: Number of image patches
    
    Returns:
        max_entropy: float — maximum possible entropy
    """
    return np.log2(n_patches)


def compute_normalized_entropy(attn_scores: torch.Tensor) -> float:
    """
    Compute normalized entropy (0 to 1 scale).
    
    0 = completely certain (all mass on one patch)
    1 = completely uncertain (uniform distribution)
    
    Args:
        attn_scores: Tensor of shape [1, n_patches]
    
    Returns:
        normalized_entropy: float in [0, 1]
    """
    n_patches = attn_scores.shape[1]
    entropy = compute_attention_entropy(attn_scores)
    max_entropy = compute_max_entropy(n_patches)
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def should_zoom(attn_scores: torch.Tensor, threshold: float = 0.75) -> bool:
    """
    Determine if zoom is needed based on attention entropy.
    
    The threshold is on normalized entropy (0-1 scale).
    Default 0.75 means: zoom when the model is >=75% as uncertain
    as it could possibly be.
    
    Recommended thresholds:
        - 0.65: Aggressive zoom (zooms on mildly uncertain cases)
        - 0.75: Balanced (default, good starting point)
        - 0.85: Conservative (only zooms on very uncertain cases)
    
    Args:
        attn_scores: Tensor of shape [1, n_patches]
        threshold: Normalized entropy threshold for triggering zoom
    
    Returns:
        bool: True if zoom should be triggered
    """
    norm_entropy = compute_normalized_entropy(attn_scores)
    return norm_entropy >= threshold


def get_top_k_patch_indices(
    attn_scores: torch.Tensor,
    n_width: int,
    n_height: int,
    k: int = 9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the top-k patches by attention score and their 2D coordinates.
    
    Args:
        attn_scores: Tensor of shape [1, n_patches]
        n_width: Number of patches in width dimension
        n_height: Number of patches in height dimension
        k: Number of top patches to return
    
    Returns:
        coords: Tensor of shape [k, 2] — (row, col) coordinates
        values: Tensor of shape [k] — attention values
    """
    k = min(k, attn_scores.shape[1])
    values, indices = torch.topk(attn_scores[0], k)
    
    rows = indices // n_width
    cols = indices % n_width
    coords = torch.stack([rows, cols], dim=1)
    
    return coords, values


# ─── Unit Tests ────────────────────────────────────────────────────────────

def test_entropy():
    """Run basic tests to verify entropy computation."""
    
    # Test 1: Uniform distribution should have max entropy
    n = 100
    uniform = torch.ones(1, n) / n
    entropy = compute_attention_entropy(uniform)
    expected = np.log2(n)
    assert abs(entropy - expected) < 1e-5, f"Uniform entropy: {entropy} != {expected}"
    
    # Test 2: Peaked distribution should have low entropy
    peaked = torch.zeros(1, n)
    peaked[0, 50] = 1.0
    entropy = compute_attention_entropy(peaked)
    assert entropy < 0.01, f"Peaked entropy should be ~0, got {entropy}"
    
    # Test 3: Normalized entropy should be in [0, 1]
    random_scores = torch.softmax(torch.randn(1, 2500), dim=-1)
    norm_ent = compute_normalized_entropy(random_scores)
    assert 0 <= norm_ent <= 1, f"Normalized entropy out of range: {norm_ent}"
    
    # Test 4: should_zoom with low threshold
    assert should_zoom(uniform, threshold=0.5) == True
    assert should_zoom(peaked, threshold=0.5) == False
    
    print("All entropy tests passed!")
    return True


if __name__ == "__main__":
    test_entropy()
