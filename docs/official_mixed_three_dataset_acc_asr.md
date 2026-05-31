# Official-Mixed Three-Dataset ACC/ASR

Snapshot date: 2026-05-31

This table reports the final strict-ASR results for NQ, HotpotQA, and MS MARCO.
ACC is clean-answer accuracy. ASR is the average attack success rate over
`poisonedrag_lm_targeted`, `poisonedrag_hotflip`, `garag`, `tan_et_al`, and
`advdecoding`.

ASR follows the strict target-hit protocol in [Strict ASR Protocol](strict_asr_protocol.md):
normalized exact match or normalized target-answer phrase containment. ASR does
not use token F1. Full per-attack results are in
[Strict-ASR Three-Dataset Results](strict_asr_three_dataset_results_20260531.md).

| Dataset | Method | ACC (%) | ASR (%) |
|---|---|---:|---:|
| NQ | Vanilla RAG | 51.80 | 9.32 |
| NQ | InstructRAG | 51.40 | 8.32 |
| NQ | AstuteRAG | 52.80 | 14.12 |
| NQ | TrustRAG | 50.80 | 3.40 |
| NQ | SeConRAG-lite | 50.80 | 3.36 |
| NQ | Learned scorer | 51.20 | 3.04 |
| NQ | Ours | 51.20 | 1.00 |
| HotpotQA | Vanilla RAG | 44.60 | 2.80 |
| HotpotQA | InstructRAG | 43.60 | 2.88 |
| HotpotQA | AstuteRAG | 44.60 | 7.20 |
| HotpotQA | TrustRAG | 43.80 | 1.28 |
| HotpotQA | SeConRAG-lite | 44.40 | 1.20 |
| HotpotQA | Learned scorer | 44.20 | 1.24 |
| HotpotQA | Ours | 42.60 | 1.00 |
| MS MARCO | Vanilla RAG | 35.20 | 1.72 |
| MS MARCO | InstructRAG | 31.00 | 1.48 |
| MS MARCO | AstuteRAG | 27.60 | 6.08 |
| MS MARCO | TrustRAG | 35.00 | 0.56 |
| MS MARCO | SeConRAG-lite | 35.20 | 0.56 |
| MS MARCO | Learned scorer | 35.00 | 0.56 |
| MS MARCO | Ours | 34.60 | 0.60 |

## Ours vs Best Baseline

| Dataset | Ours ACC (%) | Best baseline ACC (%) | Ours ASR (%) | Best baseline ASR (%) |
|---|---:|---:|---:|---:|
| NQ | 51.20 | 52.80 | 1.00 | 3.04 |
| HotpotQA | 42.60 | 44.60 | 1.00 | 1.20 |
| MS MARCO | 34.60 | 35.20 | 0.60 | 0.56 |

## Notes

- NQ and HotpotQA: ours has the lowest ASR among all methods.
- MS MARCO: ours is effectively tied with TrustRAG/SeConRAG-lite/learned scorer, with ASR 0.60% versus 0.56%.
- The main remaining cost is clean ACC, especially on HotpotQA.
