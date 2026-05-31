#!/usr/bin/env bash
set -euo pipefail

VERIRAG_ROOT="/mnt/cpfs/chenxudu/workspace/workspace_swjtu/RAG/Kimi_Agent_RAG/verirag"
OUT_BASE="/mnt/cpfs/chenxudu/workspace/workspace_swjtu/RAG/outputs/strict_asr_eval"
RUN_ROOT="${OUT_BASE}/strict_asr_three_dataset_eval_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/mnt/cpfs/chenxudu/workspace/models/Qwen3-VL-8B-Instruct"
PYTHON_BIN="/usr/local/bin/python"

mkdir -p "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${OUT_BASE}/LATEST"

cd "${VERIRAG_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=(2 3 4 5 6 7)
METHODS=(vanilla instructrag astuterag trustrag seconrag_lite learned_scorer)
DATASETS=(nq hotpotqa ms_marco)

declare -A CONFIGS
CONFIGS[nq]="configs/main/official_mixed_attack_nq500_qwen_reward_official_mixed_trained.yaml"
CONFIGS[hotpotqa]="configs/main/hotpotqa_official_mixed_qwen_reward_trained.yaml"
CONFIGS[ms_marco]="configs/main/ms_marco_official_mixed_qwen_reward_trained.yaml"

pids=()
job_index=0

next_gpu() {
  local idx=$((job_index % ${#GPUS[@]}))
  printf '%s' "${GPUS[$idx]}"
}

launch_baseline() {
  local dataset="$1"
  local method="$2"
  local gpu
  gpu="$(next_gpu)"
  job_index=$((job_index + 1))

  local job_name="baseline_${dataset}_${method}"
  local job_dir="${RUN_ROOT}/${job_name}"
  local status="${job_dir}/status.txt"
  local log="${job_dir}/eval.log"
  mkdir -p "${job_dir}"

  {
    echo "job=${job_name}"
    echo "dataset=${dataset}"
    echo "method=${method}"
    echo "gpu=${gpu}"
    echo "config=${CONFIGS[$dataset]}"
    echo "output=${job_dir}/${dataset}_${method}_baselines.md"
  } > "${status}"

  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "started=$(date -Is)" >> "${status}"
    "${PYTHON_BIN}" scripts/evaluate_rag_defense_baselines.py \
      --config "${CONFIGS[$dataset]}" \
      --dataset "${dataset}" \
      --split test \
      --n-questions 500 \
      --methods "${method}" \
      --backend transformers \
      --model-path "${MODEL_PATH}" \
      --max-new-tokens 256 \
      --temperature 0.1 \
      --top-p 0.9 \
      --output "${job_dir}/${dataset}_${method}_baselines.md" \
      > "${log}" 2>&1
    code=$?
    echo "exit_code=${code}" >> "${status}"
    echo "finished=$(date -Is)" >> "${status}"
    exit "${code}"
  ) &
  pids+=("$!")
}

launch_ours() {
  local dataset="$1"
  local gpu
  gpu="$(next_gpu)"
  job_index=$((job_index + 1))

  local job_name="ours_${dataset}"
  local job_dir="${RUN_ROOT}/${job_name}"
  local status="${job_dir}/status.txt"
  local log="${job_dir}/eval.log"
  mkdir -p "${job_dir}"

  {
    echo "job=${job_name}"
    echo "dataset=${dataset}"
    echo "method=ours"
    echo "gpu=${gpu}"
    echo "config=${CONFIGS[$dataset]}"
    echo "output=${job_dir}/${dataset}_ours.json"
    echo "trace=${job_dir}/${dataset}_ours.trace.jsonl"
  } > "${status}"

  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "started=$(date -Is)" >> "${status}"
    "${PYTHON_BIN}" scripts/diagnose_official_mixed_acc_asr.py \
      --config "${CONFIGS[$dataset]}" \
      --method ours \
      --dataset "${dataset}" \
      --split test \
      --n-questions 500 \
      --backend transformers \
      --model-path "${MODEL_PATH}" \
      --max-new-tokens 128 \
      --temperature 0.1 \
      --top-p 0.9 \
      --output "${job_dir}/${dataset}_ours.json" \
      --trace-output "${job_dir}/${dataset}_ours.trace.jsonl" \
      > "${log}" 2>&1
    code=$?
    echo "exit_code=${code}" >> "${status}"
    echo "finished=$(date -Is)" >> "${status}"
    exit "${code}"
  ) &
  pids+=("$!")
}

{
  echo "run_root=${RUN_ROOT}"
  echo "started=$(date -Is)"
  echo "gpus=${GPUS[*]}"
  echo "datasets=${DATASETS[*]}"
  echo "methods=${METHODS[*]}"
  echo "baseline_max_new_tokens=256"
  echo "ours_max_new_tokens=128"
} > "${RUN_ROOT}/manifest.txt"

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    launch_baseline "${dataset}" "${method}"
  done
done

for dataset in "${DATASETS[@]}"; do
  launch_ours "${dataset}"
done

printf '%s\n' "${pids[@]}" > "${RUN_ROOT}/pids.txt"

fail=0
for pid in "${pids[@]}"; do
  wait "${pid}" || fail=1
done

{
  echo "finished=$(date -Is)"
  echo "exit_code=${fail}"
} >> "${RUN_ROOT}/manifest.txt"

exit "${fail}"
