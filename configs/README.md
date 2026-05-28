# Config Index

## Active Mainline

- `main/official_benchmark_500_nq_doc_policy.yaml`
  Current NQ-500 multi-attack evaluation with learned scorer, NQ doc policy,
  and conflict-aware evidence control.

- `main/official_benchmark_500_nq_doc_policy_poisonedrag_only.yaml`
  Current NQ-500 PoisonedRAG-only evaluation.

- `main/nq_doc_policy_train.yaml`
  Current NQ document policy training config. Before paper submission this must
  be split into train/dev/test instead of training on the evaluation split.

## Active Baseline / Ablation

- `main/official_benchmark_500_learned_docpolicy.yaml`
- `main/official_benchmark_500_heuristic_docpolicy.yaml`
- `ablation/msmarco_learned_threshold_0.35.yaml`
- `ablation/msmarco_learned_threshold_0.45.yaml`

## Legacy

- `config.yaml`
- `main/ppo_sparse_poisonedrag_fixed_train.yaml`
- `main/sparse_poisonedrag_docpolicy.yaml`

These are retained for older experiments and compatibility, but should not be
used as the primary paper protocol.
