# VeriRAG: Verification-Guided Evidence Control for RAG Safety

> **VeriRAG** 当前主线是 verification-guided document-level RAG defense：先对每篇检索文档进行风险建模，再执行 keep/drop/abstain 证据选择，最后用 verify 信号在生成前控制证据暴露，避免 Qwen/LLM 被污染证据链带偏。

---

## 当前主线

维护入口:

- `docs/CURRENT_MAINLINE.md`: 当前论文主线、活跃代码和 legacy 边界。
- `scripts/README.md`: 训练、评估、数据对齐脚本索引。
- `configs/README.md`: 当前主配置和旧配置索引。

```text
Query + Retrieved Docs
  -> Document Risk Scorer
  -> Document-level Evidence Policy
  -> Verification-guided Evidence Controller
  -> Protected Qwen Generation / Abstain
```

这条线对应当前正式 NQ/PoisonedRAG 实验中实际起作用的模块：

- **Document Risk Scorer**: 使用 query-doc / doc-doc 相似度、rank、style、outlier、cluster、support/conflict 等真实特征评估每篇文档风险。
- **Learned Adversarial Doc Scorer**: 在启发式风险特征之上训练文档级攻击分类器，是当前最强的稳定过滤基线。
- **NQ Document Policy**: 输出文档级 keep/drop mask 和全局 abstain 动作，用于在 scorer 之上学习证据选择策略。
- **Verification-guided Evidence Controller**: 在答案生成前用 support/conflict/risk 信号复核证据，硬删高风险攻击证据，同时 rescue 高支持度 clean 证据，降低 CleanDrop。
- **Official-aligned Evaluation**: BEIR/query/doc/qrels 负责检索证据，官方 QA 文件负责答案，`eval_gold=true` 样本才计入 ACC/F1。

旧版 Claim→Verify→Answer 和 query-level PPO 模块仍保留为辅助审计/冲突信号；当前论文主线中的 verify 不是 query-level verification depth，而是生成前的 evidence verification/control。

---

## Latest Official-Mixed NQ500 Result

当前主结果使用 unified Qwen backend 和 official-mixed NQ500 fixed-attack benchmark：

- Dataset: `data_official_mixed_attack_nq500`
- Generator: `Qwen3-VL-8B-Instruct`
- Attacks: `poisonedrag_lm_targeted`, `poisonedrag_hotflip`, `garag`, `tan_et_al`, `advdecoding`
- Training split: `data_official_mixed_attack_nq_split/train`
- Test split: 500 held-out official-answer-aligned NQ questions

| Method | ACC | ASR | F1 | CleanDrop |
|---|---:|---:|---:|---:|
| Vanilla RAG | 0.5080 | 0.0956 | 0.6506 | 0.0000 |
| Learned scorer | 0.5060 | 0.0344 | 0.6640 | 0.0172 |
| SeConRAG | 0.5060 | 0.0384 | 0.6631 | 0.0136 |
| **VeriRAG verify-guided** | **0.5080** | **0.0144** | **0.6704** | **0.0140** |

CleanDrop is the fraction of clean retrieved evidence dropped by the defense. The latest verify-guided controller recovers ACC by reducing clean evidence damage while keeping ASR at oracle-level. Do not compare these numbers directly with older qrels-context NQ tables because this official-mixed setup uses PoisonedRAG official Contriever top-5 contexts.


---

## 核心模块

- `verirag/adversarial_doc_scorer.py`: 启发式文档风险评分器。
- `verirag/learned_doc_scorer.py`: 学习式文档攻击分类器。
- `verirag/nq_doc_features.py`: NQ 文档级 policy 特征构建。
- `verirag/nq_doc_policy.py`: 文档级 keep/drop/abstain policy。
- `verirag/nq_document_mask_environment.py`: 固定攻击环境下的 NQ 文档 mask 训练环境。
- `verirag/conflict_aware_generation.py`: 生成前冲突感知证据控制。
- `verirag/defense_orchestrator.py`: 防御流水线编排入口。
- `scripts/import_official_answers.py`: 官方答案导入与 coverage 统计。
- `scripts/evaluate_rag_defense_baselines.py`: NQ baseline 对齐评估。
- `scripts/evaluate.py`: 正式 VeriRAG/Qwen 评估入口。

---

## 项目架构

```
verirag/
├── configs/
│   ├── config.yaml
│   └── main/
│       ├── official_benchmark_500_nq_doc_policy.yaml
│       ├── official_benchmark_500_nq_doc_policy_poisonedrag_only.yaml
│       └── nq_doc_policy_train.yaml
├── verirag/
│   ├── adversarial_doc_scorer.py        # heuristic document risk scorer
│   ├── learned_doc_scorer.py            # learned document attack classifier
│   ├── nq_doc_features.py               # query-doc/doc-doc policy features
│   ├── nq_doc_policy.py                 # per-document keep/drop/abstain policy
│   ├── nq_document_mask_environment.py  # fixed-attack NQ policy environment
│   ├── conflict_aware_generation.py     # verification-guided evidence control
│   ├── defense_orchestrator.py          # pipeline orchestration
│   ├── claim_extractor.py               # auxiliary claim extraction
│   └── cross_validator.py               # auxiliary conflict signals
├── scripts/
│   ├── import_official_answers.py
│   ├── prepare_official_benchmark.py
│   ├── train_doc_scorer.py
│   ├── train_nq_doc_policy.py
│   ├── evaluate.py
│   └── evaluate_rag_defense_baselines.py
├── tests/
├── experiments/
├── requirements.txt
└── README.md
```

---

## 快速开始

### 环境安装

```bash
# 克隆仓库
git clone <repository-url>
cd verirag

# 创建虚拟环境
conda create -n verirag python=3.10
conda activate verirag

# 安装依赖
pip install -r requirements.txt

# 下载spaCy模型（用于NER）
python -m spacy download en_core_web_sm
```

### 数据准备

```bash
# 准备训练数据（包括4种数据类型）
python scripts/prepare_data.py \
    --config configs/config.yaml \
    --output ./data \
    --datasets nq hotpotqa \
    --phases all

# 生成攻击数据
python scripts/generate_attacks.py \
    --config configs/config.yaml \
    --output ./data/attacks \
    --dataset nq \
    --attack_types poisonedrag oneshot refinerag semantic_chameleon adaptive
```

### 训练

```bash
# 完整训练（Phase 0-2）
python scripts/train.py \
    --config configs/config.yaml \
    --phase all \
    --seed 42

# 仅PPO训练（Phase 1）
python scripts/train.py \
    --config configs/config.yaml \
    --phase single_agent

# 从checkpoint恢复
python scripts/train.py \
    --config configs/config.yaml \
    --resume checkpoints/best_model.pt \
    --phase adversarial
```

### 评估

```bash
# 完整评估
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/final_model.pt \
    --dataset nq \
    --n_questions 100 \
    --output evaluation_results.md

# 多数据集评估
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/final_model.pt \
    --output evaluation_results.md
```

### 单次防御推理

没有本地 Qwen 权重时也可以先跑完整防御链路，系统会使用轻量生成器和词法检索兜底：

```bash
python scripts/run_defense.py \
    --query "What is the revenue of Company X?" \
    --docs ./data/demo_docs.jsonl \
    --backend fallback \
    --top-k 5
```

有本地 Qwen/Qwen-8B-Chat 权重后，可切换到真实模型推理：

```bash
python scripts/run_defense.py \
    --query "What is the revenue of Company X?" \
    --docs ./data/demo_docs.jsonl \
    --model-path ./models/Qwen-8B-Chat \
    --backend vllm
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 未安装pytest时可用unittest
python -m unittest discover -s tests -v

# 运行特定测试
python -m pytest tests/test_claim_extractor.py -v
python -m pytest tests/test_policy_network.py -v
python -m pytest tests/test_defense.py -v
```

---

## 核心模块说明

### 1. Claim Extractor（声明提取器）

**文件**: `verirag/claim_extractor.py`

混合提取策略：
- **规则引擎**: 数值/时间/实体关系（100%精确，零LLM调用）
- **LLM辅助**: 复杂因果/比较声明（可选）

**关键设计**: `value`字段始终保留原始字符串，不经过embedding — 这是防御Embedding Blind Spot的核心机制。

```python
from verirag.claim_extractor import ClaimExtractor

extractor = ClaimExtractor(config={
    'rule_engine_enabled': True,
    'preserve_original_string': True,
})

claims = extractor.extract(
    documents=["The revenue was $15K."],
    doc_ids=['doc_1']
)

for claim in claims:
    print(f"Subject: {claim.subject}")
    print(f"Predicate: {claim.predicate}")
    print(f"Value: {claim.value}")  # 原始字符串 "$15K"
```

### 2. Cross Validator（交叉验证器）

**文件**: `verirag/cross_validator.py`

**核心创新**: Embedding-Independent验证

| 验证类型 | 方法 | 防御目标 |
|---------|------|---------|
| 数值比较 | 原始字符串直接比对 | $15K vs $65K (Embedding Blind Spot) |
| 实体比较 | 同义词/别名映射 | 实体替换攻击 |
| 时序比较 | 时间线一致性 | 时间篡改攻击 |
| 因果比较 | 反义词对检测 | 语义反转攻击 |

```python
from verirag.cross_validator import CrossValidator

validator = CrossValidator(config={
    'tolerance_method': 'adaptive',
    'precision_drift_threshold': 0.1,
})

report = validator.validate(claims)
print(f"Risk Score: {report.risk_score}")
print(f"Label: {report.label}")
```

### 3. Policy Network（策略网络）

**文件**: `verirag/policy_network.py`

层次化决策：
- **Level 1**: SKIP → VERIFY → REJECT
- **Level 2** (if VERIFY): LIGHT → DEEP → EXPAND

**Action Space** (5个动作):
- `0: SKIP_VERIFY` - 跳过验证（低风险查询）
- `1: LIGHT_VERIFY` - 轻量验证（快速规则检查）
- `2: DEEP_VERIFY` - 深度验证（完整交叉验证）
- `3: EXPAND_RETRIEVAL` - 扩展检索+深度验证
- `4: REJECT` - 拒绝回答（检测到攻击）

### 4. Attack Simulator（攻击模拟器）

**文件**: `verirag/attack_simulator.py`

支持5种攻击类型：
1. **PoisonedRAG** (SoC): Embedding Blind Spot攻击
2. **OneShot** (AuthChain): 单文档权威链伪造
3. **RefineRAG** (WLO): 词级别隐蔽改写
4. **Semantic Chameleon**: 双文档语义协调
5. **Adaptive**: 自适应攻击（根据防御反馈调整）

### 5. Defense Orchestrator（防御编排器）

**文件**: `verirag/defense_orchestrator.py`

四层验证流程：
```
用户查询 → 检索文档
    ↓
声明提取 (Claim Extractor)
    ↓
交叉验证 → conflict_indicators
    ↓
Policy Network 决策 → action
    ↓
执行验证动作
    ↓
四层最终验证:
    L1: Source Verification (来源可信度)
    L2: Evidence Verification (引用准确性)
    L3: Claim Verification (声明一致性)
    L4: Answer Verification (答案符合性)
    ↓
输出答案 + 验证报告
```

---

## 配置说明

配置文件路径: `configs/config.yaml`

### 关键配置项

```yaml
# 模型配置
model:
  state_dim: 512          # 状态向量维度
  action_dim: 5           # 动作空间维度
  hidden_dim: 256         # 隐藏层维度
  bert_model_name: "bert-base-uncased"

# 训练配置
training:
  lr: 3e-4                # 学习率
  gamma: 0.99             # 折扣因子
  gae_lambda: 0.95        # GAE lambda
  clip_epsilon: 0.2       # PPO裁剪系数
  batch_size: 64          # 批次大小
  total_steps: 100000     # 总训练步数

# 奖励配置
reward:
  weights:
    correctness: 0.5      # 答案正确性
    safety: 0.3           # 安全性
    efficiency: 0.15      # 效率
    verification: 0.05    # 验证准确性
  adaptive_schedule: true # 自适应权重

# 对抗进化配置
adversarial:
  num_attack_agents: 8    # 攻击Agent数量
  num_defense_agents: 8   # 防御Agent数量
  pbt_frequency: 100       # PBT频率
```

---

## 核心指标（持续更新开发ing）

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **ACC** | 清洁查询准确率 | 0.271（目标0.5+） |
| **ASR** | 攻击成功率 | 0.0168(最好) |

---

## 技术文档

- [系统架构设计](docs/architecture.md)
- [RL训练流程](docs/training_pipeline.md)
- [数据构建方案](docs/data_construction.md)
- [评估协议](docs/evaluation_protocol.md)

---

## 引用

```bibtex
@article{verirag2025,
  title={VeriRAG: Verification-Guided RAG Defense via Deep Reinforcement Learning},
  author={},
  journal={},
  year={2025}
}
```

---

## 许可证

本项目采用 MIT 许可证。

---

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
