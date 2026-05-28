#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-checkpoints/policy/ppo_sparse_poisonedrag_fixed_formal/final_model.pt}
MODEL_PATH=${MODEL_PATH:-/mnt/cpfs/chenxudu/workspace/models/Qwen3-VL-8B-Instruct}
N_QUESTIONS=${N_QUESTIONS:-500}
THRESHOLDS=${THRESHOLDS:-"0.35 0.45 0.50 0.55 0.65 0.75"}

mkdir -p configs/ablation results/ablation/msmarco_threshold_sweep

for THRESHOLD in ${THRESHOLDS}; do
  CONFIG_PATH="configs/ablation/msmarco_learned_threshold_${THRESHOLD}.yaml"
  THRESHOLD="${THRESHOLD}" CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
from pathlib import Path

import yaml

threshold = float(os.environ["THRESHOLD"])
config_path = Path(os.environ["CONFIG_PATH"])
with open("configs/main/official_benchmark_500_learned_docpolicy.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["defense"]["doc_filter_threshold"] = threshold
cfg["defense"]["doc_score_detection_threshold"] = threshold
cfg["defense"]["doc_scorer"]["threshold"] = threshold
cfg["evaluation"]["datasets"] = ["ms_marco"]
config_path.parent.mkdir(parents=True, exist_ok=True)
with open(config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  SAFE_THRESHOLD=${THRESHOLD/./p}
  CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 python scripts/evaluate.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset ms_marco \
    --n_questions "${N_QUESTIONS}" \
    --output "results/ablation/msmarco_threshold_sweep/ms_marco_${N_QUESTIONS}_threshold_${SAFE_THRESHOLD}.md" \
    --backend transformers \
    --model-path "${MODEL_PATH}" \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --top-p 1.0 \
    --seed 42 \
    --require-fixed-attacks
done

