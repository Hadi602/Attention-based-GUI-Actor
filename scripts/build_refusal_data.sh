"#!/usr/bin/env bash
# Materialize a refusal-augmented training set from a clean manifest.
# Usage: build_refusal_data.sh <clean_manifest.jsonl> <out_dir> [n_per_clean]
set -euo pipefail

CLEAN=\"${1:?Path to clean manifest jsonl}\"
OUT=\"${2:?Output directory}\"
N=\"${3:-1.0}\"

python -c \"
import json, sys
from pathlib import Path
from src.data.refusal_data_generator import (
    materialize_to_disk, PerturbationConfig
)
clean = [json.loads(l) for l in open('${CLEAN}') if l.strip()]
print(f'[refusal-data] {len(clean)} clean samples')
path = materialize_to_disk(
    clean, Path('${OUT}'), n_per_clean=float('${N}'),
    cfg=PerturbationConfig(),
)
print(f'[refusal-data] manifest written: {path}')
\"
"
