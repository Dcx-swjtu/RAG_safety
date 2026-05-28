# RL Training Pipeline

## Goal

The current RL training objective is to learn a document-level evidence selection policy that is consistent with test-time Qwen generation. The policy should keep answer-supporting clean evidence, drop poisoned evidence, and abstain only when the remaining evidence is unsafe or insufficient.

## Training Stages

### Stage 1: Build Official-Mixed Training Data

Use the official-mixed split construction pipeline:

```bash
python scripts/build_official_mixed_attack_nq_split.py \
  --output data_official_mixed_attack_nq_split
```

The generated split contains:

- `nq_train.jsonl`
- `nq_validation.jsonl`
- `nq_test.jsonl`
- split-specific attack files under `attacks/<split>/`

The held-out test split must not be used for policy/scorer training.

### Stage 2: Train the Document Risk Scorer

The learned scorer is trained on train-split clean and attack documents. It provides document-level attack probabilities and support/conflict-related features to the policy and controller.

Typical command pattern:

```bash
python scripts/train_doc_scorer.py \
  --data-dir data_official_mixed_attack_nq_split \
  --datasets nq \
  --split train \
  --attack-types poisonedrag_lm_targeted poisonedrag_hotflip garag tan_et_al advdecoding \
  --output experiments/doc_scorer/nq_official_mixed_train_learned_doc_scorer.pt
```

### Stage 3: Train the Document-level PPO Policy

The current policy is trained with a Qwen-in-loop reward backend. Qwen is used as a black-box generator/verifier inside the environment; PPO does not backpropagate through Qwen.

Main config:

```text
configs/main/nq_doc_policy_train_qwen_reward_official_mixed.yaml
```

Typical command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_nq_doc_policy.py \
  --config configs/main/nq_doc_policy_train_qwen_reward_official_mixed.yaml
```

## Environment Step

Each PPO episode follows this structure:

```text
sample query + clean docs + attack docs
  -> scorer creates document features
  -> policy outputs keep/drop mask and abstain logit
  -> environment builds filtered evidence
  -> Qwen generates/verifies answer from filtered evidence
  -> reward is computed from gold answer, target answer, clean evidence damage, and attack filtering
  -> PPO update
```

## Qwen-in-loop Reward

Files:

- `verirag/nq_document_mask_environment.py`
- `verirag/nq_doc_ppo_trainer.py`
- `verirag/generator.py`

The Qwen reward backend reduces train/test mismatch. Earlier surrogate rewards optimized proxy signals such as whether answer strings appeared in retained documents. The Qwen-in-loop reward instead evaluates the actual consequence of the policy decision under the same generator family used at test time.

Reward terms include:

- positive reward for correct clean answer generation
- penalty for matching attack target answer
- penalty for false abstain/reject
- penalty for dropping useful clean evidence
- bonus for dropping poisoned evidence
- optional verification/support consistency signal

## Checkpoints and Logs

Current official-mixed training output paths:

- Policy checkpoint: `experiments/nq_doc_policy_qwen_reward_official_mixed_checkpoints/nq_doc_policy_final.pt`
- Training summary: `experiments/nq_doc_policy_qwen_reward_official_mixed_checkpoints/train_summary.json`
- Scorer checkpoint: `experiments/doc_scorer/nq_official_mixed_train_learned_doc_scorer.pt`

These files are generated artifacts and are not committed to Git.

## Clean Training Rule

For paper results:

1. Train the scorer only on train split.
2. Train/tune policy on train and validation splits.
3. Freeze scorer, policy, and controller thresholds before final test.
4. Evaluate once on held-out official-mixed NQ500 test.

Do not train on `data_official_mixed_attack_nq500` test rows.

## Current Caveats

- Qwen-in-loop PPO is slow and has high reward variance.
- The learned scorer remains a strong baseline; the policy should be justified by ablation gains over scorer-only.
- The verification-guided controller currently provides the key ACC recovery by rescuing high-support clean evidence.
