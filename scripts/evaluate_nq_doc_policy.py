#!/usr/bin/env python3
"""Evaluate an NQ document-level policy checkpoint in the fixed attack env."""

from __future__ import annotations

import argparse
import json
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
from verirag.nq_document_mask_environment import NQDocumentMaskEnv  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NQ document-level policy")
    parser.add_argument("--config", default="configs/main/nq_doc_policy_train.yaml")
    parser.add_argument("--checkpoint", default="experiments/nq_doc_policy_checkpoints/nq_doc_policy_final.pt")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--attack-probability", type=float, default=None)
    parser.add_argument("--output", default="experiments/nq_doc_policy_eval.json")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_yaml(args.config)
    env_cfg = dict(cfg.get("environment", {}))
    if args.attack_probability is not None:
        env_cfg["attack_probability"] = args.attack_probability
    attack_types = env_cfg.get("attack_types", ["poisonedrag", "oneshot", "refinerag", "semantic_chameleon", "adaptive"])
    env = NQDocumentMaskEnv(
        data_dir=env_cfg.get("data_dir", "data_official_benchmark_500"),
        attack_types=attack_types,
        split=env_cfg.get("split", "test"),
        max_docs=int(env_cfg.get("max_docs", 10)),
        attack_probability=float(env_cfg.get("attack_probability", 0.5)),
        config=env_cfg,
    )
    env.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    policy_cfg = checkpoint.get("config", cfg.get("policy", {}))
    policy = NQDocumentActionPolicy(**policy_cfg)
    policy.load_state_dict(checkpoint.get("model_state", checkpoint.get("state_dict")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    rows = []
    with torch.no_grad():
        for _ in range(args.episodes):
            state = env.reset()
            state = {k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()}
            action = policy.select_action(state, deterministic=True)
            _, reward, _, info = env.step(action)
            verify = info.get("verify_signals", {}) or {}
            rows.append(
                {
                    "reward": float(reward),
                    "is_attacked": bool(info.get("is_attacked", False)),
                    "attack_succeeded": bool(info.get("attack_succeeded", False)),
                    "false_positive": bool(info.get("false_positive", False)),
                    "answer_correct": bool(info.get("answer_correct", False)),
                    "abstain": bool(info.get("abstain", False)),
                    "kept_ratio": float(info.get("kept_docs", 0)) / max(float(info.get("num_docs", 1)), 1.0),
                    "support_retained": float(verify.get("support_retained", 0.0)),
                    "attack_removed": float(verify.get("attack_removed", 0.0)),
                }
            )

    attacked = [r for r in rows if r["is_attacked"]]
    clean = [r for r in rows if not r["is_attacked"]]
    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "episodes": len(rows),
        "attacked_episodes": len(attacked),
        "clean_episodes": len(clean),
        "accuracy": mean([r["answer_correct"] for r in rows]),
        "attack_success_rate": mean([r["attack_succeeded"] for r in attacked]),
        "false_positive_rate": mean([r["false_positive"] for r in clean]),
        "abstain_rate": mean([r["abstain"] for r in rows]),
        "avg_reward": mean([r["reward"] for r in rows]),
        "avg_kept_ratio": mean([r["kept_ratio"] for r in rows]),
        "avg_support_retained": mean([r["support_retained"] for r in rows]),
        "avg_attack_removed_attacked": mean([r["attack_removed"] for r in attacked]),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
