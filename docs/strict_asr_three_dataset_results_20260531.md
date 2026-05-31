# Strict-ASR Three-Dataset Results

Snapshot date: 2026-05-31

Run root:
`/mnt/cpfs/chenxudu/workspace/workspace_swjtu/RAG/outputs/strict_asr_eval/strict_asr_three_dataset_eval_20260531_114150`

All 21 jobs completed with `exit_code=0`: six baselines and our main model on
NQ, HotpotQA, and MS MARCO.

ASR uses strict normalized target exact/phrase containment; ACC keeps the QA matcher.

| Dataset | Method | ACC % | ASR % | F1 % | LM-targeted ASR % | HotFlip ASR % | GARAG ASR % | TAN ASR % | AdvDecoding ASR % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ | Vanilla RAG | 51.80 | 9.32 | 65.94 | 23.40 | 22.00 | 0.00 | 0.00 | 1.20 |
| NQ | InstructRAG | 51.40 | 8.32 | 65.87 | 20.00 | 20.60 | 0.00 | 0.00 | 1.00 |
| NQ | AstuteRAG | 52.80 | 14.12 | 65.39 | 44.80 | 23.80 | 0.00 | 0.00 | 2.00 |
| NQ | TrustRAG | 50.80 | 3.40 | 66.58 | 3.00 | 12.40 | 0.00 | 0.00 | 1.60 |
| NQ | SeConRAG-lite | 50.80 | 3.36 | 66.59 | 3.00 | 12.20 | 0.00 | 0.00 | 1.60 |
| NQ | Learned scorer | 51.20 | 3.04 | 67.01 | 2.60 | 11.00 | 0.00 | 0.00 | 1.60 |
| NQ | Ours | 51.20 | 1.00 | 67.49 | 2.00 | 1.60 | 0.00 | 0.00 | 1.40 |
| HotpotQA | Vanilla RAG | 44.60 | 2.80 | 61.14 | 8.80 | 3.80 | 0.00 | 0.20 | 1.20 |
| HotpotQA | InstructRAG | 43.60 | 2.88 | 60.18 | 9.40 | 3.60 | 0.20 | 0.00 | 1.20 |
| HotpotQA | AstuteRAG | 44.60 | 7.20 | 60.25 | 28.00 | 5.60 | 0.00 | 0.00 | 2.40 |
| HotpotQA | TrustRAG | 43.80 | 1.28 | 60.68 | 3.40 | 1.60 | 0.20 | 0.00 | 1.20 |
| HotpotQA | SeConRAG-lite | 44.40 | 1.20 | 61.27 | 3.00 | 1.40 | 0.00 | 0.20 | 1.40 |
| HotpotQA | Learned scorer | 44.20 | 1.24 | 61.07 | 3.00 | 1.60 | 0.20 | 0.00 | 1.40 |
| HotpotQA | Ours | 42.60 | 1.00 | 59.57 | 2.60 | 1.20 | 0.00 | 0.00 | 1.20 |
| MS MARCO | Vanilla RAG | 35.20 | 1.72 | 51.83 | 6.00 | 1.80 | 0.00 | 0.00 | 0.80 |
| MS MARCO | InstructRAG | 31.00 | 1.48 | 47.16 | 5.00 | 1.60 | 0.00 | 0.00 | 0.80 |
| MS MARCO | AstuteRAG | 27.60 | 6.08 | 42.66 | 25.40 | 2.80 | 0.00 | 0.00 | 2.20 |
| MS MARCO | TrustRAG | 35.00 | 0.56 | 51.78 | 1.60 | 0.60 | 0.00 | 0.00 | 0.60 |
| MS MARCO | SeConRAG-lite | 35.20 | 0.56 | 51.99 | 1.60 | 0.60 | 0.00 | 0.00 | 0.60 |
| MS MARCO | Learned scorer | 35.00 | 0.56 | 51.78 | 1.60 | 0.60 | 0.00 | 0.00 | 0.60 |
| MS MARCO | Ours | 34.60 | 0.60 | 51.33 | 1.80 | 0.60 | 0.00 | 0.00 | 0.60 |
