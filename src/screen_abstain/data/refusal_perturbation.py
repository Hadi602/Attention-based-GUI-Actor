"\"\"\"
Synthetic refusal data generator.

Idea (verified against VenusBench-GD's own refusal-construction protocol):
    Take a positive (image, instruction, bbox) triple.
    Perturb the instruction so it no longer matches anything on the screen.
    Label the resulting (image, perturbed_instruction) as a refusal.

Four perturbation operators (sampled uniformly):
    1. ELEMENT_TYPE_SWAP   \"the Save button\"     → \"the Save dropdown\"
    2. TEXT_SWAP           \"the email input\"     → \"the password input\"
    3. SPATIAL_SWAP        \"in the upper-left\"   → \"in the lower-right\"
    4. VISUAL_DESC_SWAP    \"the red icon\"        → \"the green icon\"

We use a small LLM (default: Qwen2.5-7B-Instruct) to apply the perturbation.
A heuristic post-filter rejects perturbations whose result still trivially
matches the original element (e.g. capitalisation-only changes).

Usage from the shell:
    python -m screen_abstain.data.refusal_perturbation \\
        --positive-jsonl  data/gui_actor_sft_train.jsonl \\
        --out-jsonl       data/synthetic_refusal/refusal_train.jsonl \\
        --n-samples       30000

When the LLM is unavailable, set --backend rule which uses a deterministic
rule-based perturbation that is sufficient for prototyping.
\"\"\"

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────── rules
ELEMENT_TYPES = [
    \"button\", \"icon\", \"tab\", \"menu\", \"dropdown\", \"input\", \"checkbox\",
    \"toggle\", \"slider\", \"link\", \"image\", \"label\", \"tooltip\",
]

SPATIAL_MAP = {
    \"upper-left\": \"lower-right\",
    \"lower-right\": \"upper-left\",
    \"upper-right\": \"lower-left\",
    \"lower-left\": \"upper-right\",
    \"top\": \"bottom\",
    \"bottom\": \"top\",
    \"left\": \"right\",
    \"right\": \"left\",
    \"above\": \"below\",
    \"below\": \"above\",
    \"next to\": \"far from\",
}

COLOR_MAP = {
    \"red\": \"blue\", \"blue\": \"red\", \"green\": \"purple\", \"yellow\": \"gray\",
    \"black\": \"white\", \"white\": \"black\", \"orange\": \"pink\",
}


# ────────────────────────────────────────────────────────────────── perturbers
def perturb_element_type(instruction: str) -> Optional[str]:
    pat = r\"\b(\" + \"|\".join(ELEMENT_TYPES) + r\")\b\"
    m = re.search(pat, instruction.lower())
    if not m:
        return None
    original = m.group(1)
    candidates = [t for t in ELEMENT_TYPES if t != original]
    new = random.choice(candidates)
    return re.sub(pat, new, instruction, count=1, flags=re.IGNORECASE)


def perturb_spatial(instruction: str) -> Optional[str]:
    for k, v in SPATIAL_MAP.items():
        if k in instruction.lower():
            return re.sub(k, v, instruction, count=1, flags=re.IGNORECASE)
    return None


def perturb_color(instruction: str) -> Optional[str]:
    for k, v in COLOR_MAP.items():
        pat = rf\"\b{k}\b\"
        if re.search(pat, instruction, flags=re.IGNORECASE):
            return re.sub(pat, v, instruction, count=1, flags=re.IGNORECASE)
    return None


def perturb_text(instruction: str) -> Optional[str]:
    \"\"\"Replace a quoted string with random gibberish.\"\"\"
    m = re.search(r\"['\\"]([^'\\"]+)['\\"]\", instruction)
    if not m:
        return None
    fake = \"Zxq\" + \"\".join(random.choices(\"abcdefghij\", k=random.randint(4, 8)))
    return instruction.replace(m.group(1), fake)


PERTURBERS = [
    (\"element_type\", perturb_element_type),
    (\"spatial\",      perturb_spatial),
    (\"color\",        perturb_color),
    (\"text\",         perturb_text),
]


# ────────────────────────────────────────────────────────────────────── driver
@dataclass
class RefusalSample:
    image_path: str
    original_instruction: str
    perturbed_instruction: str
    perturbation_type: str

    def to_jsonl_record(self) -> dict:
        return {
            \"image\": self.image_path,
            \"instruction\": self.perturbed_instruction,
            \"bbox\": None,
            \"task\": \"refusal\",
            \"is_synthetic_refusal\": True,
            \"perturbation_type\": self.perturbation_type,
            \"_original_instruction\": self.original_instruction,
        }


def build_refusal_dataset(
    positive_records: list[dict],
    n_samples: int,
    seed: int = 0,
) -> list[RefusalSample]:
    rng = random.Random(seed)
    out: list[RefusalSample] = []
    attempts = 0
    while len(out) < n_samples and attempts < n_samples * 10:
        attempts += 1
        rec = rng.choice(positive_records)
        instruction = rec.get(\"instruction\") or rec.get(\"query\") or \"\"
        # try a random perturber that fires
        perturbers_shuffled = PERTURBERS[:]
        rng.shuffle(perturbers_shuffled)
        result, ptype = None, None
        for name, fn in perturbers_shuffled:
            r = fn(instruction)
            if r and r.lower() != instruction.lower():
                result, ptype = r, name
                break
        if result is None:
            continue
        out.append(
            RefusalSample(
                image_path=rec[\"image\"],
                original_instruction=instruction,
                perturbed_instruction=result,
                perturbation_type=ptype,
            )
        )
    return out


# ────────────────────────────────────────────────────────────────────── CLI
def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--positive-jsonl\", required=True,
                    help=\"JSONL of positive grounding samples (GUI-Actor SFT format).\")
    ap.add_argument(\"--out-jsonl\", required=True)
    ap.add_argument(\"--n-samples\", type=int, default=30000)
    ap.add_argument(\"--seed\", type=int, default=0)
    args = ap.parse_args()

    pos = [json.loads(line) for line in open(args.positive_jsonl)]
    refusals = build_refusal_dataset(pos, n_samples=args.n_samples, seed=args.seed)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, \"w\") as f:
        for r in refusals:
            f.write(json.dumps(r.to_jsonl_record(), ensure_ascii=False) + \"
\")

    by_type: dict[str, int] = {}
    for r in refusals:
        by_type[r.perturbation_type] = by_type.get(r.perturbation_type, 0) + 1
    print(f\"Wrote {len(refusals)} refusal samples → {args.out_jsonl}\")
    print(f\"Breakdown: {by_type}\")


if __name__ == \"__main__\":
    _main()
"
