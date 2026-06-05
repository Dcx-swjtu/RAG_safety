# VeriRAG: Verification-Guided Evidence Control for RAG Safety

VeriRAG is a research prototype for safer retrieval-augmented generation (RAG).
The current paper-facing mainline is **verification-guided document-level
evidence control**: score retrieved documents, decide which evidence to expose,
and verify evidence before generation so that a downstream Qwen/LLM generator is
less likely to follow poisoned context.

```text
Query + Retrieved Docs
  -> Document Risk Scorer
  -> Document-level Evidence Policy
  -> Verification-guided Evidence Controller
  -> Protected Qwen Generation / Abstain
```

The older query-level PPO, simulator, and Claim -> Verify -> Answer components
remain in the repository for compatibility and auxiliary signals, but they are
not the current main experiment path. See
[`docs/CURRENT_MAINLINE.md`](docs/CURRENT_MAINLINE.md) for the active/legacy
boundary.

## Highlights

- **Document-level risk scoring** with query-document, document-document, rank,
  style, outlier, cluster, support, and conflict features.
- **Learned adversarial document scorer** as a stable filtering baseline on top
  of deterministic risk features.
- **NQ document policy** that predicts per-document keep/drop masks and a
  global abstain action.
- **Verification-guided evidence controller** that removes high-risk attack
  evidence while rescuing high-support clean evidence to reduce CleanDrop.
- **Official-aligned evaluation** where BEIR/qrels provide retrieval evidence
  and official QA files provide answer supervision; only `eval_gold=true`
  samples are counted for ACC/F1.

## Latest Results

### Official-Mixed NQ500

The current main benchmark uses 500 held-out official-answer-aligned NQ
questions with unified Qwen generation and mixed official-code attacks.

- Dataset: `data_official_mixed_attack_nq500`
- Train/dev/test split: `data_official_mixed_attack_nq_split`
- Generator: `Qwen3-VL-8B-Instruct`
- Attacks: `poisonedrag_lm_targeted`, `poisonedrag_hotflip`, `garag`,
  `tan_et_al`, `advdecoding`
- Main config:
  `configs/main/official_mixed_attack_nq500_qwen_reward_official_mixed_trained.yaml`

| Method | ACC | ASR | F1 | FPR | CleanDrop |
|---|---:|---:|---:|---:|---:|
| Vanilla RAG | 0.5080 | 0.0956 | 0.6506 | 0.0000 | 0.0000 |
| InstructRAG-style | 0.5140 | 0.0928 | 0.6562 | 0.0000 | 0.0000 |
| AstuteRAG-style | 0.5200 | 0.1096 | 0.6566 | 0.0000 | 0.0000 |
| TrustRAG-style | 0.5060 | 0.0380 | 0.6632 | 0.0000 | 0.0140 |
| SeConRAG-lite | 0.5060 | 0.0384 | 0.6631 | 0.0000 | 0.0136 |
| Learned scorer | 0.5060 | 0.0344 | 0.6640 | 0.0000 | 0.0172 |
| Old Ours full | 0.4900 | 0.0160 | 0.6542 | 0.0000 | 0.1024 |
| **VeriRAG verify-guided** | **0.5080** | **0.0144** | **0.6704** | **0.0000** | **0.0140** |
| Oracle filtering reference | 0.5080 | 0.0148 | 0.6703 | 0.0000 | 0.0000 |

CleanDrop is the fraction of clean retrieved evidence removed by the defense.
It is not the same as hard rejection FPR. The latest controller improves over
the older full pipeline mainly by reducing clean evidence damage while keeping
ASR close to the oracle filtering reference.

### Three-Dataset Strict-ASR Snapshot

Snapshot date: 2026-05-31. ACC is clean-answer accuracy; ASR is the average
strict target-hit rate over the five fixed attack families.

| Dataset | Ours ACC (%) | Best Baseline ACC (%) | Ours ASR (%) | Best Baseline ASR (%) |
|---|---:|---:|---:|---:|
| NQ | 51.20 | 52.80 | 1.00 | 3.04 |
| HotpotQA | 42.60 | 44.60 | 1.00 | 1.20 |
| MS MARCO | 34.60 | 35.20 | 0.60 | 0.56 |

Full tables are available in
[`docs/official_mixed_three_dataset_acc_asr.md`](docs/official_mixed_three_dataset_acc_asr.md)
and
[`docs/strict_asr_three_dataset_results_20260531.md`](docs/strict_asr_three_dataset_results_20260531.md).

## Repository Layout

```text
.
├── configs/
│   ├── README.md
│   ├── config.yaml                         # legacy config
│   ├── main/                               # active experiment configs
│   └── ablation/                           # ablation configs
├── docs/                                   # architecture, protocols, results
├── scripts/
│   ├── README.md                           # script index
│   ├── train_doc_scorer.py
│   ├── train_nq_doc_policy.py
│   ├── evaluate.py
│   └── diagnose_official_mixed_acc_asr.py
├── tests/
├── third_party/
├── verirag/
│   ├── adversarial_doc_scorer.py
│   ├── learned_doc_scorer.py
│   ├── nq_doc_features.py
│   ├── nq_doc_policy.py
│   ├── nq_document_mask_environment.py
│   ├── conflict_aware_generation.py
│   ├── defense_orchestrator.py
│   └── generator.py
├── EXPERIMENTS.md
├── requirements.txt
└── setup.py
```

Generated datasets, checkpoints, logs, model weights, and experiment outputs are
not tracked by Git. The `.gitignore` excludes paths such as `data/`, `data_*`,
`experiments/`, `logs/`, `results/`, `checkpoints/`, and `models/`.

## Installation

```bash
git clone https://github.com/Dcx-swjtu/RAG_safety.git
cd RAG_safety

conda create -n verirag python=3.10
conda activate verirag

pip install -r requirements.txt
pip install -e ".[dev]"
```

Optional components:

```bash
# spaCy model for modules that use NER
python -m spacy download en_core_web_sm

# Optional Qwen/vLLM support
pip install -e ".[qwen]"
```

## Data and Model Prerequisites

The main evaluation expects local copies of the official-mixed benchmark data
and a local Qwen model path. These large artifacts are intentionally excluded
from Git.

Expected local data paths for the current mainline:

- `data_official_mixed_attack_nq500`
- `data_official_mixed_attack_nq_split`

Expected model:

- `Qwen3-VL-8B-Instruct` or another compatible local model path passed through
  `--model-path`

## Quick Start

Run tests:

```bash
python -m pytest tests -v
```

Train the learned document scorer:

```bash
python scripts/train_doc_scorer.py \
  --data-dir data_official_mixed_attack_nq_split \
  --datasets nq \
  --splits train \
  --output experiments/doc_scorer/learned_doc_scorer.pt
```

Train the official-mixed Qwen-in-loop NQ document policy:

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
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --output outputs/official_mixed_nq500_ours.json
```

Run the general VeriRAG evaluator:

```bash
python scripts/evaluate.py \
  --config configs/main/official_mixed_attack_nq500_qwen_reward_official_mixed_trained.yaml \
  --dataset nq \
  --n_questions 500 \
  --backend transformers \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --output evaluation_results.md
```

## Active Entry Points

| Purpose | File |
|---|---|
| Mainline definition | [`docs/CURRENT_MAINLINE.md`](docs/CURRENT_MAINLINE.md) |
| Experiment notes | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Script index | [`scripts/README.md`](scripts/README.md) |
| Config index | [`configs/README.md`](configs/README.md) |
| Document risk scorer | [`verirag/adversarial_doc_scorer.py`](verirag/adversarial_doc_scorer.py) |
| Learned doc scorer | [`verirag/learned_doc_scorer.py`](verirag/learned_doc_scorer.py) |
| NQ doc policy | [`verirag/nq_doc_policy.py`](verirag/nq_doc_policy.py) |
| Evidence controller | [`verirag/conflict_aware_generation.py`](verirag/conflict_aware_generation.py) |
| Pipeline orchestrator | [`verirag/defense_orchestrator.py`](verirag/defense_orchestrator.py) |
| Qwen/fallback backend | [`verirag/generator.py`](verirag/generator.py) |

## Technical Documents

- [System Architecture](docs/architecture.md)
- [Training Pipeline](docs/training_pipeline.md)
- [Data Construction](docs/data_construction.md)
- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Strict ASR Protocol](docs/strict_asr_protocol.md)
- [NQ Train / Dev / Test Protocol](docs/NQ_TRAIN_DEV_TEST_PROTOCOL.md)

## Legacy Components

The following files are retained for old experiments, tests, and compatibility
interfaces, but should not be used as the primary paper story:

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

## Citation

```bibtex
@article{verirag2025,
  title={VeriRAG: Verification-Guided Evidence Control for RAG Safety},
  author={},
  journal={},
  year={2025}
}
```

## License

No license file is currently included in this repository. Add a `LICENSE` file
before distributing or reusing the code outside the project.
