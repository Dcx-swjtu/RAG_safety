# Current Mainline

This file defines the code path that should be treated as the active research
prototype. Older query-level PPO and simulator code is kept for reference, but
is not the paper-facing mainline.

## Paper-Facing Pipeline

```text
Query + Retrieved Docs
  -> document risk scoring
  -> document-level evidence policy
  -> conflict-aware evidence control
  -> protected generation / abstain
```

## Active Package Files

- `verirag/adversarial_doc_scorer.py`: heuristic document risk scoring.
- `verirag/learned_doc_scorer.py`: learned adversarial document classifier.
- `verirag/text_features.py`: deterministic query/document features.
- `verirag/nq_doc_features.py`: NQ document policy feature builder.
- `verirag/nq_doc_policy.py`: per-document keep/drop/abstain policy.
- `verirag/nq_document_mask_environment.py`: fixed-attack NQ policy environment.
- `verirag/nq_doc_ppo_trainer.py`: NQ document policy trainer.
- `verirag/conflict_aware_generation.py`: generation-time evidence control.
- `verirag/defense_orchestrator.py`: compatibility pipeline entry point.
- `verirag/generator.py`: Qwen/fallback generation backend.

## Active Scripts

- `scripts/import_official_answers.py`: import official NQ/HotpotQA/MS MARCO answers.
- `scripts/import_sparse_poisonedrag_attacks.py`: import paper PoisonedRAG-style attacks.
- `scripts/prepare_official_benchmark.py`: build official-aligned fixed-attack benchmark.
- `scripts/train_doc_scorer.py`: train document attack scorer.
- `scripts/train_nq_doc_policy.py`: train NQ document-level policy.
- `scripts/evaluate.py`: evaluate the full VeriRAG defense pipeline.
- `scripts/evaluate_nq_doc_policy.py`: evaluate NQ policy environment checkpoints.
- `scripts/evaluate_rag_defense_baselines.py`: unified local baseline comparison.

## Active Configs

- `configs/main/official_benchmark_500_nq_doc_policy.yaml`: current NQ-500 main config.
- `configs/main/official_benchmark_500_nq_doc_policy_poisonedrag_only.yaml`: PoisonedRAG-only main config.
- `configs/main/nq_doc_policy_train.yaml`: current NQ policy training config.

## Legacy / Auxiliary Code

These files are still importable because tests and compatibility entry points
refer to them, but they should not be used as the primary paper story:

- `verirag/attack_simulator.py`
- `verirag/environment.py`
- `verirag/fixed_attack_environment.py`
- `verirag/policy_network.py`
- `verirag/ppo_trainer.py`
- `verirag/reward_function.py`
- `verirag/state_encoder.py`
- `scripts/train.py`
- `scripts/generate_attacks.py`
- `scripts/prepare_data.py`
- `configs/config.yaml`

## Current Evidence To Watch

The current Qwen NQ-500 run writes to:

- `experiments/official_benchmark_500_nq_doc_policy_conflict_aware_qwen500_eval.log`
- `experiments/official_benchmark_500_nq_doc_policy_conflict_aware_qwen500_eval.md`

Do not move these files or their source config while that run is active.
