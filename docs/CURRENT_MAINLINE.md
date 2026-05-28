# Current Mainline

This file defines the code path that should be treated as the active research
prototype. Older query-level PPO and simulator code is kept for reference, but
is not the paper-facing mainline.

## Paper-Facing Pipeline

```text
Query + Retrieved Docs
  -> document risk scoring
  -> document-level evidence policy
  -> verification-guided evidence control
  -> protected Qwen generation / abstain
```

The current method should be described as verification-guided evidence control.
The verify signal is used to control which evidence reaches the generator, not
as a query-level choice of verification depth.

## Active Package Files

- `verirag/adversarial_doc_scorer.py`: heuristic document risk scoring.
- `verirag/learned_doc_scorer.py`: learned adversarial document classifier.
- `verirag/text_features.py`: deterministic query/document features.
- `verirag/nq_doc_features.py`: NQ document policy feature builder.
- `verirag/nq_doc_policy.py`: per-document keep/drop/abstain policy.
- `verirag/nq_document_mask_environment.py`: fixed-attack NQ policy environment with optional Qwen-in-loop reward.
- `verirag/nq_doc_ppo_trainer.py`: NQ document policy trainer.
- `verirag/conflict_aware_generation.py`: verification-guided generation-time evidence control.
- `verirag/defense_orchestrator.py`: compatibility pipeline entry point.
- `verirag/generator.py`: Qwen/fallback generation backend.

## Active Scripts

- `scripts/import_official_answers.py`: import official NQ/HotpotQA/MS MARCO answers.
- `scripts/build_official_mixed_attack_nq500.py`: build the held-out official-mixed NQ500 benchmark.
- `scripts/build_official_mixed_attack_nq_split.py`: build train/validation/test splits for official-mixed training.
- `scripts/prepare_official_benchmark.py`: build official-aligned fixed-attack benchmark.
- `scripts/train_doc_scorer.py`: train document attack scorer.
- `scripts/train_nq_doc_policy.py`: train NQ document-level policy.
- `scripts/evaluate.py`: evaluate the full VeriRAG defense pipeline.
- `scripts/diagnose_official_mixed_acc_asr.py`: official-mixed ACC/ASR/CleanDrop diagnostics.
- `scripts/evaluate_rag_defense_baselines.py`: unified local baseline comparison.

## Active Configs

- `configs/main/official_mixed_attack_nq500_qwen_reward_official_mixed_trained.yaml`: current official-mixed main config.
- `configs/main/nq_doc_policy_train_qwen_reward_official_mixed.yaml`: Qwen-in-loop policy training config.
- `configs/main/nq_doc_policy_train_official_mixed.yaml`: surrogate policy training config on official-mixed data.
- `configs/main/official_benchmark_500_nq_doc_policy.yaml`: older NQ-500 qrels-context config.

## Latest Official-Mixed NQ500 Evidence

All runs use the unified Qwen backend on 500 held-out official-answer-aligned NQ
questions with mixed official-code attack implementations. The latest
verification-guided controller is the current main result.

| Method | ACC | ASR | F1 | CleanDrop |
|---|---:|---:|---:|---:|
| Vanilla RAG | 0.5080 | 0.0956 | 0.6506 | 0.0000 |
| Learned scorer | 0.5060 | 0.0344 | 0.6640 | 0.0172 |
| SeConRAG-lite | 0.5060 | 0.0384 | 0.6631 | 0.0136 |
| Old Ours full | 0.4900 | 0.0160 | 0.6542 | 0.1024 |
| VeriRAG verify-guided | 0.5080 | 0.0144 | 0.6704 | 0.0140 |
| Oracle filtering reference | 0.5080 | 0.0148 | 0.6703 | 0.0000 |

CleanDrop is clean evidence damage, not hard rejection FPR. The current main
controller improves over the old full pipeline primarily by rescuing high-support
clean evidence while preserving very low ASR.

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
