#!/usr/bin/env python3
"""Train the NQ-only document-level keep/drop policy."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verirag.nq_doc_policy import NQDocumentActionPolicy  # noqa: E402
from verirag.nq_doc_ppo_trainer import NQDocPPOTrainer  # noqa: E402
from verirag.nq_document_mask_environment import NQDocumentMaskEnv  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NQ document-level policy")
    parser.add_argument("--config", default="configs/main/nq_doc_policy_train.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)

    env_cfg = cfg.get("environment", {})
    attack_types = env_cfg.get(
        "attack_types",
        ["poisonedrag", "oneshot", "refinerag", "semantic_chameleon", "adaptive"],
    )
    env = NQDocumentMaskEnv(
        data_dir=env_cfg.get("data_dir", "data_official_benchmark_500"),
        attack_types=attack_types,
        split=env_cfg.get("split", "test"),
        max_docs=int(env_cfg.get("max_docs", 10)),
        attack_probability=float(env_cfg.get("attack_probability", 0.5)),
        config=env_cfg,
    )
    env.seed(seed)

    policy_cfg = cfg.get("policy", {})
    policy = NQDocumentActionPolicy(
        input_dim=int(policy_cfg.get("input_dim", env.feature_dim)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 256)),
        doc_state_dim=int(policy_cfg.get("doc_state_dim", 128)),
        global_dim=int(policy_cfg.get("global_dim", 256)),
        dropout=float(policy_cfg.get("dropout", 0.1)),
    )

    trainer = NQDocPPOTrainer(policy, env, cfg.get("training", {}))
    qwen_reward_cfg = env_cfg.get("qwen_reward", {}) or {}
    print(
        f"[NQDocTrain] data={env.data_dir} split={env.split} samples={len(env.samples)} "
        f"attackable={len(env.attackable_samples)} feature_dim={env.feature_dim} seed={seed} "
        f"qwen_reward={bool(qwen_reward_cfg.get('enabled', False))}"
    )
    summary = trainer.train()
    print(f"[NQDocTrain] final_checkpoint={summary['final_checkpoint']}")


if __name__ == "__main__":
    main()
