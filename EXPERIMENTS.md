# VeriRAG Experiment Layout

This project now keeps reproducible experiment assets under stable top-level
directories while preserving the original `experiments/` outputs.

## Directory Map

- `configs/main/`: canonical configs for main training and evaluation runs.
- `configs/ablation/`: configs for controlled ablation and threshold sweeps.
- `results/main/`: canonical result reports copied from completed runs.
- `results/ablation/`: reserved for scorer-only, PPO-only, verify ablations, and threshold sweeps.
- `checkpoints/policy/`: stable links to policy checkpoints.
- `checkpoints/doc_scorer/`: stable links to learned document scorer checkpoints.
- `scripts/reproduce/`: reproducible command wrappers for main experiments.

Original files remain in `experiments/` for backward compatibility.

## Main Checkpoints

- PPO policy:
  `checkpoints/policy/ppo_sparse_poisonedrag_fixed_formal/final_model.pt`
- Learned doc scorer:
  `checkpoints/doc_scorer/learned_doc_scorer.pt`

## Main Official Benchmark 500 Results

All runs use `data_official_benchmark_500`, fixed attacks, local Qwen backend,
and 500 gold/scorable samples per dataset.

| Dataset | Scorer | ACC | ASR | F1 | FPR | Notes |
|---|---|---:|---:|---:|---:|---|
| NQ | heuristic | 0.8500 | 0.0928 | 0.8777 | 0.0400 | Strong utility, higher PoisonedRAG ASR |
| NQ | learned | 0.8200 | 0.0420 | 0.8836 | 0.0960 | Best NQ F1, lower ASR |
| HotpotQA | heuristic | 0.5460 | 0.2248 | 0.6407 | 0.0160 | Conservative false positives |
| HotpotQA | learned | 0.7200 | 0.1776 | 0.7678 | 0.1420 | Best HotpotQA overall |
| MS MARCO | heuristic | 0.2540 | 0.1632 | 0.3897 | 0.0920 | Lower ASR than learned |
| MS MARCO | learned | 0.2720 | 0.1888 | 0.4074 | 0.0640 | Slightly better utility/F1, worse ASR |

Canonical reports:

- `results/main/official_benchmark_500/heuristic_docpolicy/`
- `results/main/official_benchmark_500/learned_docpolicy/`
- `results/main/sparse_poisonedrag/docpolicy/`

## Reproduction

Train the learned document scorer:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_doc_scorer.py \
  --data-dir data_official_benchmark_500 \
  --datasets nq hotpotqa ms_marco \
  --attack-types poisonedrag oneshot refinerag semantic_chameleon adaptive \
  --output experiments/doc_scorer/learned_doc_scorer.pt \
  --epochs 6 \
  --batch-size 256 \
  --max-clean-docs-per-sample 10 \
  --max-attack-docs-per-sample 10
```

Run the official benchmark with the learned scorer:

```bash
bash scripts/reproduce/run_official_benchmark_500_learned_gpu4_7.sh
```

Run the official benchmark with the heuristic scorer:

```bash
bash scripts/reproduce/run_official_benchmark_500_heuristic_gpu4_7.sh
```

## Current Risks

- The learned scorer was trained and evaluated on the same benchmark family.
  Add held-out or cross-attack splits before making final SOTA claims.
- MS MARCO remains the weakest dataset. The learned scorer improves ACC/FPR
  slightly but raises ASR, so threshold sweep is required.
- PPO and verify contributions need explicit ablations:
  scorer-only, PPO-only, scorer+PPO, scorer+PPO+verify.
- Current official configs still use lightweight/fallback text encoders unless a
  local embedding model is configured.

