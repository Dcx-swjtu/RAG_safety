# VeriRAG Experiment Layout

This project keeps source code, configuration, and reproducibility notes in Git.
Generated datasets, experiment logs, checkpoints, and model weights are excluded
from the repository.

## Current Main Experiment

The current paper-facing setup is the official-mixed NQ500 benchmark with unified
Qwen generation. It combines held-out official-answer-aligned NQ questions with
mixed official-code attack implementations.

- Dataset directory: `data_official_mixed_attack_nq500`
- Train/dev/test directory: `data_official_mixed_attack_nq_split`
- Generator: `Qwen3-VL-8B-Instruct`
- Test questions: 500 held-out NQ samples
- Attacks: `poisonedrag_lm_targeted`, `poisonedrag_hotflip`, `garag`, `tan_et_al`, `advdecoding`
- Main config: `configs/main/official_mixed_attack_nq500_qwen_reward_official_mixed_trained.yaml`

## Main Official-Mixed NQ500 Results

| Method | ACC | ASR | F1 | FPR | CleanDrop |
|---|---:|---:|---:|---:|---:|
| Vanilla RAG | 0.5080 | 0.0956 | 0.6506 | 0.0000 | 0.0000 |
| InstructRAG-style | 0.5140 | 0.0928 | 0.6562 | 0.0000 | 0.0000 |
| AstuteRAG-style | 0.5200 | 0.1096 | 0.6566 | 0.0000 | 0.0000 |
| TrustRAG-style | 0.5060 | 0.0380 | 0.6632 | 0.0000 | 0.0140 |
| SeConRAG-lite | 0.5060 | 0.0384 | 0.6631 | 0.0000 | 0.0136 |
| Learned scorer | 0.5060 | 0.0344 | 0.6640 | 0.0000 | 0.0172 |
| Old Ours full | 0.4900 | 0.0160 | 0.6542 | 0.0000 | 0.1024 |
| VeriRAG verify-guided | 0.5080 | 0.0144 | 0.6704 | 0.0000 | 0.0140 |
| Oracle filtering reference | 0.5080 | 0.0148 | 0.6703 | 0.0000 | 0.0000 |

## Attack ASR Breakdown

| Method | LM-targeted | HotFlip | GARAG | Tan et al. | AdvDecoding |
|---|---:|---:|---:|---:|---:|
| Vanilla RAG | 0.2180 | 0.2180 | 0.0120 | 0.0160 | 0.0140 |
| Learned scorer | 0.0260 | 0.1080 | 0.0120 | 0.0120 | 0.0140 |
| SeConRAG-lite | 0.0280 | 0.1240 | 0.0120 | 0.0120 | 0.0160 |
| VeriRAG verify-guided | 0.0180 | 0.0160 | 0.0120 | 0.0120 | 0.0140 |

## Interpretation

- The current model's main advantage is ASR reduction on targeted PoisonedRAG-style attacks, especially HotFlip.
- The previous full policy had strong ASR but excessive CleanDrop. The latest verification-guided controller keeps ASR at oracle-level while recovering ACC.
- CleanDrop is not FPR. It measures clean retrieved evidence removed by the defense; high CleanDrop can silently reduce clean ACC even when no query is rejected.
- `*-style` baselines are unified-Qwen local wrappers unless a separate official-repo wrapper is explicitly reported.

## Reproduction Entrypoints

Train the official-mixed Qwen-in-loop policy:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_nq_doc_policy.py \
  --config configs/main/nq_doc_policy_train_qwen_reward_official_mixed.yaml
```

Run the main official-mixed diagnostic evaluation:

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

## Current Risks

- Official-repo baseline wrappers should be tracked separately from local style baselines.
- The main innovation claim should be verification-guided evidence control, not query-level PPO.
- Future ablations should isolate scorer-only, policy-only, controller-only, and Qwen-in-loop reward effects.
