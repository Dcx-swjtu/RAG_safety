#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} python scripts/train_doc_scorer.py \
  --data-dir data_official_benchmark_500 \
  --datasets nq hotpotqa ms_marco \
  --attack-types poisonedrag oneshot refinerag semantic_chameleon adaptive \
  --output experiments/doc_scorer/learned_doc_scorer.pt \
  --epochs 6 \
  --batch-size 256 \
  --max-clean-docs-per-sample 10 \
  --max-attack-docs-per-sample 10

