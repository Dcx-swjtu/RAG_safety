# Script Index

Use this file as the stable entry-point map. Scripts not listed as active are
kept for backward compatibility or old ablations.

## Active

### Data Alignment

- `import_official_answers.py`
- `import_sparse_poisonedrag_attacks.py`
- `prepare_beir_data.py`
- `prepare_aligned_eval_data.py`
- `prepare_official_benchmark.py`

### Training

- `train_doc_scorer.py`
- `train_nq_doc_policy.py`

### Evaluation

- `evaluate.py`
- `evaluate_nq_doc_policy.py`
- `evaluate_rag_defense_baselines.py`

### Reproduction Shell Wrappers

- `reproduce/train_learned_doc_scorer.sh`
- `reproduce/run_nq_doc_policy_qwen500_gpu4_7.sh`
- `reproduce/run_official_benchmark_500_learned_gpu4_7.sh`
- `reproduce/run_official_benchmark_500_heuristic_gpu4_7.sh`

## Legacy / Auxiliary

- `train.py`: old query-level PPO training path.
- `generate_attacks.py`: simulator-driven attacks; not the main fixed-attack path.
- `prepare_data.py`: older synthetic/combined data preparation path.
- `run_defense.py`: single-query demo entry point.

Keep legacy scripts available until the paper-facing train/eval protocol is
fully frozen.
