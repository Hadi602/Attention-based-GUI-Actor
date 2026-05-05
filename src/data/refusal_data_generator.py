"\"\"\"
Perturbation-based refusal data generator.

Given a clean GUI grounding sample (image, query, target_bbox), produce
a \"refusable\" counterpart by applying one of several perturbations
that make the target absent or the instruction unsatisfiable.

Perturbations
-------------
1. crop_out_target
   Crop the screenshot to a region that excludes the target bbox.
   Instruction is kept; the referenced element is no longer visible.

2. mask_target
   Inpaint / blur the target bbox so the element is no longer
   visually identifiable. (Cheap variant: fill with mean colour.)

3. swap_screenshot
   Pair the instruction with a *different* screenshot (random sample
   from the dataset). High-confidence refusal example.

4. instruction_substitution
   Replace the referent in the instruction with a non-existent element
   (\"Click the Bluetooth toggle\" on a screen that has none). Requires
   a small pool of plausible-but-absent referents per app/domain.

5. resolution_zoom
   Zoom into a non-target region; useful for the resolution-invariant
   ablation (backup direction).

Output schema
-------------
Each generated sample is a dict:
    {
        \"image\": PIL.Image,
        \"instruction\": str,
        \"target_bbox\": None,            # always None for refusal
        \"is_refusal\": True,
        \"perturbation\": str,            # which strategy produced this
        \"source_id\": str,               # provenance for debugging
    }

Author: Hadi
Project: ScreenAbstain (2026)
\"\"\"

from __future__ import annotations
import random
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable

from PIL import Image, ImageFilter


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #
@dataclass
class PerturbationConfig:
    p_crop_out: float = 0.30
    p_mask_target: float = 0.20
    p_swap_screenshot: float = 0.30
    p_instruction_sub: float = 0.15
    p_resolution_zoom: float = 0.05

    # crop_out_target
    crop_padding: int = 20            # px buffer around removed region
    min_crop_area_frac: float = 0.25  # keep ≥ 25% of original area

    # mask_target
    mask_mode: str = \"blur\"           # \"blur\" | \"mean\" | \"noise\"
    mask_blur_radius: int = 25

    # instruction_substitution
    distractor_vocab: List[str] = field(default_factory=lambda: [
        \"Bluetooth toggle\", \"VPN switch\", \"airplane mode button\",
        \"dark mode toggle\", \"barcode scanner\", \"QR code button\",
        \"voice assistant icon\", \"screen recorder\",
    ])

    seed: int = 42


# ---------------------------------------------------------------------- #
# Generator
# ---------------------------------------------------------------------- #
class RefusalDataGenerator:
    \"\"\"Stateless-ish generator: pass clean samples in, get refusal samples out.\"\"\"

    def __init__(self, cfg: Optional[PerturbationConfig] = None) -> None:
        self.cfg = cfg or PerturbationConfig()
        self.rng = random.Random(self.cfg.seed)
        self._strategies = self._build_strategy_table()

    # ------------------------------------------------------------------ #
    def _build_strategy_table(self):
        c = self.cfg
        return [
            (\"crop_out_target\",       c.p_crop_out,        self._crop_out_target),
            (\"mask_target\",           c.p_mask_target,     self._mask_target),
            (\"swap_screenshot\",       c.p_swap_screenshot, self._swap_screenshot),
            (\"instruction_sub\",       c.p_instruction_sub, self._instruction_sub),
            (\"resolution_zoom\",       c.p_resolution_zoom, self._resolution_zoom),
        ]

    def _pick_strategy(self):
        names, probs, fns = zip(*self._strategies)
        return self.rng.choices(list(zip(names, fns)), weights=probs, k=1)[0]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(
        self,
        clean_sample: Dict,
        sample_pool: Optional[List[Dict]] = None,
    ) -> Optional[Dict]:
        \"\"\"
        Args
        ----
        clean_sample : {image, instruction, target_bbox, source_id}
        sample_pool  : list of *other* clean samples (needed for swap)

        Returns
        -------
        Refusal sample dict or None if perturbation failed.
        \"\"\"
        name, fn = self._pick_strategy()
        try:
            return fn(clean_sample, sample_pool)
        except Exception as exc:                                    # noqa: BLE001
            print(f\"[refusal-gen] {name} failed for \"
                  f\"{clean_sample.get('source_id','?')}: {exc}\")
            return None

    def generate_batch(
        self,
        clean_samples: List[Dict],
        n_refusal: Optional[int] = None,
    ) -> List[Dict]:
        \"\"\"Produce a batch of refusal samples sized n_refusal (default = len).\"\"\"
        n_refusal = n_refusal or len(clean_samples)
        out: List[Dict] = []
        for _ in range(n_refusal):
            src = self.rng.choice(clean_samples)
            ref = self.generate(src, sample_pool=clean_samples)
            if ref is not None:
                out.append(ref)
        return out

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #
    def _crop_out_target(self, s, _pool):
        img: Image.Image = s[\"image\"]
        bbox = s[\"target_bbox\"]                   # (x1, y1, x2, y2)
        W, H = img.size
        x1, y1, x2, y2 = bbox
        pad = self.cfg.crop_padding

        # Choose the largest of 4 candidate crops that excludes the bbox.
        candidates = [
            (0, 0, max(1, x1 - pad), H),                     # left strip
            (min(W - 1, x2 + pad), 0, W, H),                 # right strip
            (0, 0, W, max(1, y1 - pad)),                     # top strip
            (0, min(H - 1, y2 + pad), W, H),                 # bottom strip
        ]
        candidates = [c for c in candidates
                      if (c[2] - c[0]) > 0 and (c[3] - c[1]) > 0]
        if not candidates:
            return None
        # Pick by area, prefer ≥ min_crop_area_frac of original.
        candidates.sort(key=lambda c: (c[2]-c[0])*(c[3]-c[1]), reverse=True)
        crop = candidates[0]
        if (crop[2]-crop[0])*(crop[3]-crop[1]) < \
           self.cfg.min_crop_area_frac * W * H:
            return None
        new_img = img.crop(crop)
        return {
            \"image\": new_img,
            \"instruction\": s[\"instruction\"],
            \"target_bbox\": None,
            \"is_refusal\": True,
            \"perturbation\": \"crop_out_target\",
            \"source_id\": s.get(\"source_id\", \"\"),
        }

    def _mask_target(self, s, _pool):
        img: Image.Image = s[\"image\"].copy()
        x1, y1, x2, y2 = s[\"target_bbox\"]
        region = img.crop((x1, y1, x2, y2))
        if self.cfg.mask_mode == \"blur\":
            region = region.filter(
                ImageFilter.GaussianBlur(radius=self.cfg.mask_blur_radius)
            )
        elif self.cfg.mask_mode == \"mean\":
            mean = tuple(int(c) for c in region.resize((1, 1)).getpixel((0, 0)))
            region = Image.new(img.mode, region.size, mean)
        else:  # noise
            import numpy as np
            arr = (255 * np.random.rand(region.size[1], region.size[0], 3)
                   ).astype(\"uint8\")
            region = Image.fromarray(arr)
        img.paste(region, (x1, y1))
        return {
            \"image\": img,
            \"instruction\": s[\"instruction\"],
            \"target_bbox\": None,
            \"is_refusal\": True,
            \"perturbation\": \"mask_target\",
            \"source_id\": s.get(\"source_id\", \"\"),
        }

    def _swap_screenshot(self, s, pool):
        if not pool or len(pool) < 2:
            return None
        other = self.rng.choice(pool)
        # Cheap collision check: skip if same source_id
        if other.get(\"source_id\") == s.get(\"source_id\"):
            return None
        return {
            \"image\": other[\"image\"],
            \"instruction\": s[\"instruction\"],
            \"target_bbox\": None,
            \"is_refusal\": True,
            \"perturbation\": \"swap_screenshot\",
            \"source_id\": f\"{s.get('source_id','')}__on__{other.get('source_id','')}\",
        }

    def _instruction_sub(self, s, _pool):
        distractor = self.rng.choice(self.cfg.distractor_vocab)
        new_instr = f\"Click the {distractor}.\"
        return {
            \"image\": s[\"image\"],
            \"instruction\": new_instr,
            \"target_bbox\": None,
            \"is_refusal\": True,
            \"perturbation\": \"instruction_sub\",
            \"source_id\": s.get(\"source_id\", \"\"),
        }

    def _resolution_zoom(self, s, _pool):
        img: Image.Image = s[\"image\"]
        bbox = s[\"target_bbox\"]
        W, H = img.size
        x1, y1, x2, y2 = bbox

        # Zoom into a corner that does NOT contain the target.
        corners = {
            \"tl\": (0, 0, W // 2, H // 2),
            \"tr\": (W // 2, 0, W, H // 2),
            \"bl\": (0, H // 2, W // 2, H),
            \"br\": (W // 2, H // 2, W, H),
        }
        valid = [c for c in corners.values()
                 if not _bbox_overlaps(c, bbox)]
        if not valid:
            return None
        crop = self.rng.choice(valid)
        new_img = img.crop(crop).resize((W, H))
        return {
            \"image\": new_img,
            \"instruction\": s[\"instruction\"],
            \"target_bbox\": None,
            \"is_refusal\": True,
            \"perturbation\": \"resolution_zoom\",
            \"source_id\": s.get(\"source_id\", \"\"),
        }


def _bbox_overlaps(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


# ---------------------------------------------------------------------- #
# CLI: generate-and-dump (manifest-driven)
# ---------------------------------------------------------------------- #
def materialize_to_disk(
    clean_manifest: List[Dict],
    out_dir: Path,
    n_per_clean: float = 1.0,
    cfg: Optional[PerturbationConfig] = None,
) -> Path:
    \"\"\"
    clean_manifest : list of {image_path, instruction, target_bbox, source_id}
    Writes:
        out_dir/images/*.png
        out_dir/refusal_manifest.jsonl
    Returns path to manifest.
    \"\"\"
    out_dir = Path(out_dir)
    (out_dir / \"images\").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / \"refusal_manifest.jsonl\"

    # Load images lazily into memory-light dicts
    pool = []
    for m in clean_manifest:
        pool.append({
            \"image\": Image.open(m[\"image_path\"]).convert(\"RGB\"),
            \"instruction\": m[\"instruction\"],
            \"target_bbox\": tuple(m[\"target_bbox\"]),
            \"source_id\": m[\"source_id\"],
        })

    gen = RefusalDataGenerator(cfg)
    n_refusal = int(len(pool) * n_per_clean)
    refusals = gen.generate_batch(pool, n_refusal=n_refusal)

    with open(manifest_path, \"w\") as f:
        for i, r in enumerate(refusals):
            img_path = out_dir / \"images\" / f\"refusal_{i:06d}.png\"
            r[\"image\"].save(img_path)
            f.write(json.dumps({
                \"image_path\": str(img_path),
                \"instruction\": r[\"instruction\"],
                \"is_refusal\": True,
                \"perturbation\": r[\"perturbation\"],
                \"source_id\": r[\"source_id\"],
            }) + \"
\")
    return manifest_path
"
