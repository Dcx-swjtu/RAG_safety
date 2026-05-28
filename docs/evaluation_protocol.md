# Evaluation Protocol

## Current Benchmark

The current main benchmark is official-mixed NQ500:

- Dataset: `data_official_mixed_attack_nq500`
- Test split: 500 held-out official-answer-aligned NQ questions
- Generator: Qwen3-VL-8B-Instruct
- Attacks: `poisonedrag_lm_targeted`, `poisonedrag_hotflip`, `garag`, `tan_et_al`, `advdecoding`
- Main config: `configs/main/official_mixed_attack_nq500_qwen_reward_official_mixed_trained.yaml`

## Main Evaluation Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/diagnose_official_mixed_acc_asr.py \
  --config configs/main/official_mixed_attack_nq500_qwen_reward_official_mixed_trained.yaml \
  --method ours \
  --dataset nq \
  --split test \
  --n-questions 500 \
  --backend transformers \
  --model-path /path/to/Qwen3-VL-8B-Instruct
```

## Metrics

| Metric | Direction | Definition |
|---|---:|---|
| ACC | higher is better | clean-query answer accuracy against official gold answers |
| ASR | lower is better | attack success rate, measured by matching attack target answer |
| F1 | higher is better | token-level answer F1 on official-gold clean questions |
| FPR | lower is better | hard false positive rejection rate on clean questions |
| CleanDrop | lower is better | fraction of clean retrieved evidence dropped by the defense |

CleanDrop is not the same as FPR. A model may have FPR 0 but still reduce ACC by silently dropping clean support evidence.

## Main Table Format

| Method | ACC | ASR | F1 | FPR | CleanDrop |
|---|---:|---:|---:|---:|---:|
| Vanilla RAG | 0.5060 | 0.0940 | 0.6493 | 0.0000 | 0.0000 |
| InstructRAG | 0.5140 | 0.0928 | 0.6562 | 0.0000 | 0.0000 |
| AstuteRAG | 0.5200 | 0.1096 | 0.6566 | 0.0000 | 0.0000 |
| TrustRAG | 0.5060 | 0.0380 | 0.6632 | 0.0000 | 0.0140 |
| SeCon-RAG | 0.5040 | 0.0380 | 0.6615 | 0.0000 | 0.0136 |
| Learned Scorer | 0.5040 | 0.0340 | 0.6624 | 0.0000 | 0.0172 |
| Ours | 0.5080 | 0.0144 | 0.6704 | 0.0000 | 0.0140 |

## Attack Breakdown Format

| Method | LM-targeted | HotFlip | GARAG | Tan et al. | AdvDecoding |
|---|---:|---:|---:|---:|---:|
| Vanilla RAG | 0.2060 | 0.2200 | 0.0120 | 0.0160 | 0.0160 |
| InstructRAG | 0.1940 | 0.2080 | 0.0220 | 0.0220 | 0.0180 |
| AstuteRAG | 0.3140 | 0.2080 | 0.0020 | 0.0020 | 0.0220 |
| TrustRAG | 0.0280 | 0.1220 | 0.0120 | 0.0120 | 0.0160 |
| SeCon-RAG | 0.0280 | 0.1220 | 0.0120 | 0.0120 | 0.0160 |
| Learned Scorer | 0.0260 | 0.1060 | 0.0120 | 0.0120 | 0.0140 |
| Ours | 0.0180 | 0.0160 | 0.0120 | 0.0120 | 0.0140 |

## Baseline Naming

Use the original method names in paper tables:

- InstructRAG
- AstuteRAG
- TrustRAG
- SeCon-RAG
- Learned Scorer
- Vanilla RAG

Do not append `style` or `lite` in the main table when reporting the current official-code-aligned runs. If a result is only a prompt-style surrogate, state that caveat in the text or appendix instead.

## Result Files

Generated experiment outputs are not tracked by Git. In the local workspace, the relevant result files are:

- `experiments/official_mixed_experiment_summary_20260528.md`
- `experiments/official_mixed_experiment_summary_20260528.json`
- `experiments/official_mixed_baseline_repro_20260528/*.json`
- `experiments/verification_guided_main_20260528_174550/ours_verification_guided_nq500.json`

## Interpretation Rules

- Do not compare official-mixed NQ500 ACC directly with older qrels-context tables. The context source differs.
- Treat oracle filtering as a reference line, not a real deployable method.
- Claims should emphasize low ASR with recovered ACC, not just lowest ASR.
- Always report CleanDrop when discussing clean accuracy loss.
- Keep generator, decoding settings, and test split fixed across all methods.
