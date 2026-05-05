"\"\"\"
Loader for VenusBench-GD (arXiv 2512.16501).

The benchmark publishes images and a JSONL annotation file with fields:
    {
        \"image\": \"path/to/screenshot.png\",
        \"instruction\": \"...\",
        \"bbox\": [x1, y1, x2, y2]   # normalised, OR null for refusal samples
        \"task\": \"element\" | \"spatial\" | \"visual\" | \"reasoning\" | \"functional\" | \"refusal\",
        \"language\": \"en\" | \"zh\",
        \"platform\": \"web\" | \"mobile\" | \"desktop\",
        \"application\": \"Photoshop\",
    }

Once the dataset URL is finalised, set VENUSBENCH_DIR in `scripts/00_setup.sh`
to point at the local copy.
\"\"\"

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class VenusSample:
    sample_id: str
    image_path: str
    instruction: str
    bbox: Optional[List[float]]    # None ↔ refusal
    task: str
    language: str
    platform: str
    application: str

    @property
    def is_refusal(self) -> bool:
        return self.bbox is None or self.task == \"refusal\"


def load_venusbench(
    annotations_path: str | Path,
    images_dir: str | Path,
    task_filter: Optional[List[str]] = None,
    language_filter: Optional[List[str]] = None,
) -> List[VenusSample]:
    \"\"\"
    Load VenusBench-GD samples.

    Args:
        annotations_path: path to `annotations.json` or `annotations.jsonl`.
        images_dir: directory containing the screenshots referenced from
            `annotations[i][\"image\"]`.
        task_filter: optionally restrict to a subset, e.g. [\"refusal\"].
        language_filter: optionally restrict to [\"en\"] or [\"zh\"].

    Returns:
        list of VenusSample
    \"\"\"
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)

    raw: list[dict]
    if annotations_path.suffix == \".jsonl\":
        raw = [json.loads(line) for line in annotations_path.open()]
    else:
        with open(annotations_path) as f:
            raw = json.load(f)

    samples: list[VenusSample] = []
    for i, r in enumerate(raw):
        if task_filter is not None and r.get(\"task\") not in task_filter:
            continue
        if language_filter is not None and r.get(\"language\") not in language_filter:
            continue
        samples.append(
            VenusSample(
                sample_id=r.get(\"id\", f\"venus_{i:06d}\"),
                image_path=str(images_dir / r[\"image\"]),
                instruction=r[\"instruction\"],
                bbox=r.get(\"bbox\"),
                task=r.get(\"task\", \"element\"),
                language=r.get(\"language\", \"en\"),
                platform=r.get(\"platform\", \"unknown\"),
                application=r.get(\"application\", \"unknown\"),
            )
        )
    return samples
"
