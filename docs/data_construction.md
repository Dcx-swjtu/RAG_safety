# Data Construction Plan

## Purpose

The dataset is designed to evaluate RAG defense under a controlled but realistic evidence-poisoning setting. The key requirement is to preserve official QA answers for clean accuracy while constructing fixed mixed attacks that stress document-level verification.

## Data Sources

The local source data is stored outside Git under the workspace data directories. The Git repository only tracks construction scripts and documentation.

Main sources:

- BEIR-style NQ query/document/qrels files
- official NQ-open answers
- official or official-code-aligned attack implementations collected under the local `RAG/data_process` and `RAG/baselines` directories

## Construction Stages

### Stage 1: Official Answer Alignment

Script:

```text
scripts/import_official_answers.py
```

Purpose:

- BEIR provides query/document/qrels structure.
- Official QA files provide answer lists.
- Rows with official answers are marked `eval_gold=true`.
- Only `eval_gold=true` rows are used for final ACC/F1 scoring.

### Stage 2: Official-Gold Benchmark Selection

Script:

```text
scripts/prepare_official_benchmark.py
```

Purpose:

- Keep official-gold rows.
- Build benchmark subsets.
- Generate or attach fixed attack files in a reproducible layout.

### Stage 3: Official-Mixed NQ500 Test Construction

Script:

```text
scripts/build_official_mixed_attack_nq500.py
```

Main output:

```text
data_official_mixed_attack_nq500/
  nq_test.jsonl
  official_mixed_attack_manifest.json
  attacks/
    test/
      nq_poisonedrag_lm_targeted.jsonl
      nq_poisonedrag_hotflip.jsonl
      nq_garag.jsonl
      nq_tan_et_al.jsonl
      nq_advdecoding.jsonl
```

The current held-out test set contains 500 official-answer-aligned NQ questions.

### Stage 4: Train / Validation / Test Split Construction

Script:

```text
scripts/build_official_mixed_attack_nq_split.py
```

Main output:

```text
data_official_mixed_attack_nq_split/
  nq_train.jsonl
  nq_validation.jsonl
  nq_test.jsonl
  official_mixed_attack_split_manifest.json
  attacks/
    train/
    validation/
    test/
```

The training split is used for scorer and policy training. The test split is kept held out.

## Attack Types

Current official-mixed benchmark uses five attack families:

| Attack | Meaning |
|---|---|
| `poisonedrag_lm_targeted` | PoisonedRAG-style LM-targeted poisoned evidence |
| `poisonedrag_hotflip` | PoisonedRAG-style HotFlip poisoned evidence |
| `garag` | GARAG-style generated adversarial retrieval evidence |
| `tan_et_al` | Tan et al.-style knowledge poisoning evidence |
| `advdecoding` | adversarial decoding based attack evidence |

The current benchmark is mixed-attack by design. It is intended to test whether defense can handle different poisoned evidence patterns under one unified generator and metric protocol.

## Why This Fits VeriRAG

VeriRAG is a verification-guided evidence control method. The dataset therefore focuses on whether the system can decide which retrieved documents should reach the generator.

The benchmark exposes these failure modes:

- poisoned document is retained and causes target answer generation
- clean support evidence is wrongly dropped
- evidence set becomes insufficient after filtering
- generator answers incorrectly despite low attack exposure
- hard rejection/FPR and silent clean evidence damage diverge

This is why the evaluation tracks both ASR and CleanDrop.

## What Is Not Committed

The following are not stored in Git:

- generated JSONL datasets
- official answer files
- attack payload files
- checkpoints
- Qwen weights
- raw experiment logs

Only scripts, configs, tests, and documentation are committed.

## Reproducibility Notes

For a clean paper run:

1. Rebuild official-mixed data with the tracked scripts.
2. Train scorer and policy only on the train split.
3. Tune on validation split.
4. Report final numbers on held-out NQ500 test.
5. Use the same Qwen backend and decoding settings for all methods.
