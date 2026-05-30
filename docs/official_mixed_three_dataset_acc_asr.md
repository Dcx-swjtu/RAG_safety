# Official-Mixed Three-Dataset ACC/ASR

Snapshot date: 2026-05-30

This table reports the current formal results for NQ, HotpotQA, and MS MARCO.
ACC is clean-answer accuracy. ASR is the average attack success rate over
`poisonedrag_lm_targeted`, `poisonedrag_hotflip`, `garag`, `tan_et_al`, and
`advdecoding`.

| Dataset | Method | ACC (%) | ASR (%) |
|---|---|---:|---:|
| NQ | Vanilla RAG | 50.60 | 9.40 |
| NQ | InstructRAG | 51.40 | 9.28 |
| NQ | AstuteRAG | 52.00 | 10.96 |
| NQ | TrustRAG | 50.60 | 3.80 |
| NQ | SeConRAG-lite | 50.40 | 3.80 |
| NQ | Learned scorer | 50.40 | 3.40 |
| NQ | Ours | 50.80 | 1.44 |
| HotpotQA | Vanilla RAG | 44.20 | 4.64 |
| HotpotQA | InstructRAG | 44.40 | 4.84 |
| HotpotQA | AstuteRAG | 44.60 | 7.40 |
| HotpotQA | TrustRAG | 44.40 | 4.48 |
| HotpotQA | SeConRAG-lite | 44.80 | 4.76 |
| HotpotQA | Learned scorer | 44.00 | 4.80 |
| HotpotQA | Ours | 43.00 | 2.56 |
| MS MARCO | Vanilla RAG | 34.80 | 7.20 |
| MS MARCO | InstructRAG | 31.80 | 6.20 |
| MS MARCO | AstuteRAG | 27.80 | 7.64 |
| MS MARCO | TrustRAG | 34.80 | 7.20 |
| MS MARCO | SeConRAG-lite | 34.80 | 7.04 |
| MS MARCO | Learned scorer | 34.80 | 7.16 |
| MS MARCO | Ours | 32.80 | 5.40 |

## Ours vs Best Baseline

| Dataset | Ours ACC (%) | Best baseline ACC (%) | Ours ASR (%) | Best baseline ASR (%) |
|---|---:|---:|---:|---:|
| NQ | 50.80 | 52.00 | 1.44 | 3.40 |
| HotpotQA | 43.00 | 44.80 | 2.56 | 4.48 |
| MS MARCO | 32.80 | 34.80 | 5.40 | 6.20 |
