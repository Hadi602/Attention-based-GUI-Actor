"""
Attention based Zoom Grounding — Zoom Module
=======================================
Handles cropping, resizing, and coordinate remapping for the
adaptive zoom inference pipeline.

Author: Rashid (Attention based Zoom Grounding Project)
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class ZoomRegion:
    """Represents a crop region in normalized [0,1] coordinates."""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = max(0.0, x1)
        self.y1 = max(0.0, y1)
        self.x2 = min(1.0, x2)
        self.y2 = min(1.0, y2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_pixel_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coords to pixel coordinates."""
        return (
            int(self.x1 * img_width),
            int(self.y1 * img_height),
            int(self.x2 * img_width),
            int(self.y2 * img_height)
        )
    
    def __repr__(self):
        return f"ZoomRegion(x1={self.x1:.3f}, y1={self.y1:.3f}, x2={self.x2:.3f}, y2={self.y2:.3f})"


def compute_zoom_region(
    top_k_coords: torch.Tensor,
    n_width: int,
    n_height: int,
    padding_ratio: float = 0.15,
    min_region_size: float = 0.2
) -> ZoomRegion:
    """
    Compute the zoom region from top-k patch coordinates.
    
    Takes the bounding box of the top-k attended patches and adds
    padding to create a stable crop region.
    
    Args:
        top_k_coords: Tensor of shape [k, 2] — (row, col) patch coordinates
        n_width: Number of patches in width dimension
        n_height: Number of patches in height dimension
        padding_ratio: Extra padding around the bounding box (fraction of region size)
        min_region_size: Minimum region size as fraction of full image (prevents too-tight crops)
    
    Returns:
        ZoomRegion: The computed crop region in normalized coordinates
    """
    # Convert patch coordinates to normalized [0,1] coordinates
    rows = top_k_coords[:, 0].float()
    cols = top_k_coords[:, 1].float()
    
    # Patch boundaries in normalized space
    y1 = (rows.min() / n_height).item()
    y2 = ((rows.max() + 1) / n_height).item()
    x1 = (cols.min() / n_width).item()
    x2 = ((cols.max() + 1) / n_width).item()
    
    # Compute region dimensions
    width = x2 - x1
    height = y2 - y1
    
    # Ensure minimum size
    if width < min_region_size:
        center_x = (x1 + x2) / 2
        x1 = center_x - min_region_size / 2
        x2 = center_x + min_region_size / 2
        width = min_region_size
    
    if height < min_region_size:
        center_y = (y1 + y2) / 2
        y1 = center_y - min_region_size / 2
        y2 = center_y + min_region_size / 2
        height = min_region_size
    
    # Add padding
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y
    
    return ZoomRegion(x1, y1, x2, y2)


def crop_and_resize(
    image: Image.Image,
    region: ZoomRegion,
    target_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Crop the image to the zoom region and resize to target size.
    
    If no target_size is provided, uses the original image dimensions
    (effectively zooming in to fill the same canvas).
    
    Args:
        image: PIL Image to crop
        region: ZoomRegion specifying the crop area
        target_size: Optional (width, height) to resize to. If None, uses original size.
    
    Returns:
        Cropped and resized PIL Image
    """
    img_width, img_height = image.size
    
    # Convert to pixel coordinates
    px1, py1, px2, py2 = region.to_pixel_coords(img_width, img_height)
    
    # Ensure valid crop dimensions
    px1 = max(0, px1)
    py1 = max(0, py1)
    px2 = min(img_width, px2)
    py2 = min(img_height, py2)
    
    # Crop
    cropped = image.crop((px1, py1, px2, py2))
    
    # Resize to target (or original) dimensions
    if target_size is None:
        target_size = (img_width, img_height)
    
    resized = cropped.resize(target_size, Image.LANCZOS)
    
    return resized


def remap_coordinates(
    pred_x: float,
    pred_y: float,
    region: ZoomRegion
) -> Tuple[float, float]:
    """
    Remap predicted coordinates from crop space back to original image space.
    
    The model predicts coordinates in [0,1] relative to the cropped image.
    This function maps them back to the full image coordinate system.
    
    Args:
        pred_x: Predicted x coordinate in crop space [0, 1]
        pred_y: Predicted y coordinate in crop space [0, 1]
        region: The ZoomRegion used for cropping
    
    Returns:
        (orig_x, orig_y): Coordinates in original image space [0, 1]
    """
    # Linear mapping from crop space to original space
    orig_x = region.x1 + pred_x * region.width
    orig_y = region.y1 + pred_y * region.height
    
    # Clamp to valid range
    orig_x = max(0.0, min(1.0, orig_x))
    orig_y = max(0.0, min(1.0, orig_y))
    
    return orig_x, orig_y


def compute_effective_patch_size(
    original_size: Tuple[int, int],
    region: ZoomRegion,
    base_patch_size: int = 28
) -> Tuple[float, float]:
    """
    Compute the effective patch size after zooming.
    
    This tells you how many original pixels each patch covers
    after the zoom operation. Smaller = better resolution.
    
    Args:
        original_size: (width, height) of original image
        region: The ZoomRegion
        base_patch_size: The model's native patch size (28 for Qwen2-VL)
    
    Returns:
        (eff_width, eff_height): Effective patch size in original pixels
    """
    img_width, img_height = original_size
    
    # The zoom region covers this many pixels
    region_width_px = region.width * img_width
    region_height_px = region.height * img_height
    
    # After resize, the model still uses the same patch size
    # But each patch now covers fewer original pixels
    # Number of patches = image_size / patch_size (approximately)
    # So effective patch size = region_size / num_patches_in_that_dim
    
    # Since we resize the crop to full image size, the number of patches
    # remains the same. Each patch now covers:
    eff_width = (region_width_px / img_width) * base_patch_size
    eff_height = (region_height_px / img_height) * base_patch_size
    
    return eff_width, eff_height


# ─── Unit Tests ────────────────────────────────────────────────────────────

def test_zoom():
    """Run basic tests for zoom utilities."""
    
    # Test 1: ZoomRegion clamping
    region = ZoomRegion(-0.1, -0.1, 1.1, 1.1)
    assert region.x1 == 0.0 and region.y1 == 0.0
    assert region.x2 == 1.0 and region.y2 == 1.0
    
    # Test 2: Coordinate remapping
    region = ZoomRegion(0.25, 0.25, 0.75, 0.75)
    # Center of crop should map to center of region
    ox, oy = remap_coordinates(0.5, 0.5, region)
    assert abs(ox - 0.5) < 1e-5 and abs(oy - 0.5) < 1e-5
    
    # Top-left of crop should map to top-left of region
    ox, oy = remap_coordinates(0.0, 0.0, region)
    assert abs(ox - 0.25) < 1e-5 and abs(oy - 0.25) < 1e-5
    
    # Test 3: Effective patch size
    eff_w, eff_h = compute_effective_patch_size((1920, 1080), region)
    # Region is 50% of image, so effective patch = 50% of 28 = 14
    assert abs(eff_w - 14.0) < 1e-5
    assert abs(eff_h - 14.0) < 1e-5
    
    # Test 4: compute_zoom_region
    coords = torch.tensor([[5, 10], [5, 11], [6, 10], [6, 11]])
    region = compute_zoom_region(coords, n_width=40, n_height=30)
    assert region.x1 >= 0 and region.y1 >= 0
    assert region.x2 <= 1 and region.y2 <= 1
    
    print("All zoom tests passed!")
    return True


if __name__ == "__main__":
    test_zoom()
