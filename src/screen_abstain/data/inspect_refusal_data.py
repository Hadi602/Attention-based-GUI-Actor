"\"\"\"Quick visual inspection of the synthetic refusal dataset.\"\"\"
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--jsonl\",
                    default=\"data/synthetic_refusal/refusal_train.jsonl\")
    ap.add_argument(\"--n\", type=int, default=20)
    args = ap.parse_args()

    if not Path(args.jsonl).exists():
        raise SystemExit(f\"Not found: {args.jsonl}. Did you run scripts/02 first?\")

    records = [json.loads(line) for line in open(args.jsonl)]
    random.shuffle(records)
    for r in records[: args.n]:
        print(\"─\" * 72)
        print(f\"image:                {r['image']}\")
        print(f\"perturbation_type:    {r.get('perturbation_type')}\")
        print(f\"original instruction: {r.get('_original_instruction')}\")
        print(f\"perturbed instruction:{r['instruction']}\")


if __name__ == \"__main__\":
    _main()
"
