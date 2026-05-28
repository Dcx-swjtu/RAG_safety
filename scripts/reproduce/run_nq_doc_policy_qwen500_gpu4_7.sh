#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/evaluate.py   --config configs/main/official_benchmark_500_nq_doc_policy.yaml   --checkpoint experiments/ppo_sparse_poisonedrag_fixed_formal_checkpoints/best_model.pt   --dataset nq   --n_questions 500   --output experiments/official_benchmark_500_nq_doc_policy_qwen500_eval.md   --backend transformers   --model-path /mnt/cpfs/chenxudu/workspace/models/Qwen3-VL-8B-Instruct   --max-new-tokens 256   --temperature 0.1   --top-p 0.9   --require-fixed-attacks
