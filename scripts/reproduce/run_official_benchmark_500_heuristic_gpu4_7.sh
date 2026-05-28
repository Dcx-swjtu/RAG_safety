#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-checkpoints/policy/ppo_sparse_poisonedrag_fixed_formal/final_model.pt}
MODEL_PATH=${MODEL_PATH:-/mnt/cpfs/chenxudu/workspace/models/Qwen3-VL-8B-Instruct}
N_QUESTIONS=${N_QUESTIONS:-500}

for DATASET in nq hotpotqa ms_marco; do
  CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 python scripts/evaluate.py \
    --config configs/main/official_benchmark_500_heuristic_docpolicy.yaml \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --n_questions "${N_QUESTIONS}" \
    --output "results/main/official_benchmark_500/heuristic_docpolicy/${DATASET}_${N_QUESTIONS}.md" \
    --backend transformers \
    --model-path "${MODEL_PATH}" \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --top-p 1.0 \
    --seed 42 \
    --require-fixed-attacks
done

