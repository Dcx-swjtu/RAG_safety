"""PPO trainer for the NQ document-level keep/drop policy."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .nq_doc_policy import NQDocumentActionPolicy
from .nq_document_mask_environment import NQDocumentMaskEnv


@dataclass
class NQDocTransition:
    doc_features: torch.Tensor
    doc_mask: torch.Tensor
    keep_action: torch.Tensor
    abstain_action: torch.Tensor
    reward: float
    value: float
    log_prob: float
    done: bool


class NQDocRolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.clear()

    def clear(self) -> None:
        self.transitions: List[NQDocTransition] = []

    @property
    def size(self) -> int:
        return len(self.transitions)

    def store(self, state: Dict[str, torch.Tensor], action: Dict[str, torch.Tensor], reward: float, done: bool) -> None:
        if self.size >= self.capacity:
            return
        self.transitions.append(
            NQDocTransition(
                doc_features=state["doc_features"].detach().cpu().squeeze(0),
                doc_mask=state["doc_mask"].detach().cpu().squeeze(0),
                keep_action=action["keep_mask"].detach().cpu().squeeze(0),
                abstain_action=action["abstain"].detach().cpu().view(1),
                reward=float(reward),
                value=float(action["value"].detach().cpu().view(-1)[0]),
                log_prob=float(action["log_prob"].detach().cpu().view(-1)[0]),
                done=bool(done),
            )
        )

    def returns_and_advantages(self, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = np.asarray([t.reward for t in self.transitions], dtype=np.float32)
        values = np.asarray([t.value for t in self.transitions] + [0.0], dtype=np.float32)
        dones = np.asarray([t.done for t in self.transitions], dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for idx in reversed(range(len(rewards))):
            next_value = 0.0 if dones[idx] else values[idx + 1]
            delta = rewards[idx] + gamma * next_value - values[idx]
            last_gae = delta if dones[idx] else delta + gamma * gae_lambda * last_gae
            advantages[idx] = last_gae
        returns = advantages + values[:-1]
        return torch.from_numpy(returns), torch.from_numpy(advantages)

    def batches(self, batch_size: int, returns: torch.Tensor, advantages: torch.Tensor, epochs: int):
        indices = np.arange(self.size)
        for _ in range(int(epochs)):
            np.random.shuffle(indices)
            for start in range(0, self.size, int(batch_size)):
                batch_idx = indices[start:start + int(batch_size)]
                yield {
                    "doc_features": torch.stack([self.transitions[i].doc_features for i in batch_idx]),
                    "doc_mask": torch.stack([self.transitions[i].doc_mask for i in batch_idx]),
                    "keep_actions": torch.stack([self.transitions[i].keep_action for i in batch_idx]),
                    "abstain_actions": torch.stack([self.transitions[i].abstain_action for i in batch_idx]).view(-1),
                    "old_log_probs": torch.tensor([self.transitions[i].log_prob for i in batch_idx], dtype=torch.float32),
                    "returns": returns[batch_idx].float(),
                    "advantages": advantages[batch_idx].float(),
                }


class NQDocPPOTrainer:
    """Small PPO loop for one-step NQ document masking episodes."""

    def __init__(
        self,
        policy: NQDocumentActionPolicy,
        env: NQDocumentMaskEnv,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.policy = policy
        self.env = env
        self.config = config or {}
        self.device = torch.device(self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy.to(self.device)

        self.lr = float(self.config.get("lr", 3e-4))
        self.gamma = float(self.config.get("gamma", 0.99))
        self.gae_lambda = float(self.config.get("gae_lambda", 0.95))
        self.clip_epsilon = float(self.config.get("clip_epsilon", 0.2))
        self.entropy_coef = float(self.config.get("entropy_coef", 0.01))
        self.value_coef = float(self.config.get("value_coef", 0.5))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 0.5))
        self.rollout_length = int(self.config.get("rollout_length", 128))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.epochs_per_update = int(self.config.get("epochs_per_update", 4))
        self.total_steps = int(self.config.get("total_steps", 5000))
        self.log_freq = int(self.config.get("log_freq", 100))
        self.checkpoint_freq = int(self.config.get("checkpoint_freq", 1000))
        self.checkpoint_dir = str(self.config.get("checkpoint_dir", "experiments/nq_doc_policy_checkpoints"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.lr, weight_decay=float(self.config.get("weight_decay", 1e-4)))
        self.buffer = NQDocRolloutBuffer(self.rollout_length)
        self.step_count = 0
        self.update_count = 0
        self.history: List[Dict[str, float]] = []

    def _to_device(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in state.items()}

    def collect_rollout(self) -> Dict[str, float]:
        self.buffer.clear()
        stats = {
            "reward": [],
            "attack_success": [],
            "false_positive": [],
            "answer_correct": [],
            "kept_ratio": [],
            "attack_removed": [],
            "support_retained": [],
            "abstain": [],
            "qwen_reward_calls": [],
            "qwen_answer_correct": [],
            "qwen_attack_success": [],
            "qwen_reward_delta": [],
        }
        self.policy.eval()
        while self.buffer.size < self.rollout_length and self.step_count < self.total_steps:
            state = self._to_device(self.env.reset())
            action = self.policy.select_action(state, deterministic=False)
            _, reward, done, info = self.env.step(action)
            self.buffer.store(state, action, reward, done)
            self.step_count += 1

            n_docs = max(int(info.get("num_docs", 1)), 1)
            verify = info.get("verify_signals", {}) or {}
            qwen_reward = info.get("qwen_reward", {}) or {}
            stats["reward"].append(float(reward))
            stats["attack_success"].append(float(info.get("attack_succeeded", False)))
            stats["false_positive"].append(float(info.get("false_positive", False)))
            stats["answer_correct"].append(float(info.get("answer_correct", False)))
            stats["kept_ratio"].append(float(info.get("kept_docs", 0)) / n_docs)
            stats["attack_removed"].append(float(verify.get("attack_removed", 0.0)))
            stats["support_retained"].append(float(verify.get("support_retained", 0.0)))
            stats["abstain"].append(float(info.get("abstain", False)))
            stats["qwen_reward_calls"].append(float(qwen_reward.get("enabled", False) and not qwen_reward.get("skipped", False)))
            stats["qwen_answer_correct"].append(float(qwen_reward.get("answer_correct", False)))
            stats["qwen_attack_success"].append(float(qwen_reward.get("attack_succeeded", False)))
            stats["qwen_reward_delta"].append(float(qwen_reward.get("reward_delta", 0.0)))
        return {key: float(np.mean(values)) if values else 0.0 for key, values in stats.items()}

    def update(self) -> Dict[str, float]:
        returns, advantages = self.buffer.returns_and_advantages(self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
        losses: Dict[str, List[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []}
        self.policy.train()

        for batch in self.buffer.batches(self.batch_size, returns, advantages, self.epochs_per_update):
            inputs = {
                "doc_features": batch["doc_features"].to(self.device),
                "doc_mask": batch["doc_mask"].to(self.device),
            }
            keep_actions = batch["keep_actions"].to(self.device)
            abstain_actions = batch["abstain_actions"].to(self.device)
            old_log_probs = batch["old_log_probs"].to(self.device)
            return_batch = batch["returns"].to(self.device)
            advantage_batch = batch["advantages"].to(self.device)

            out = self.policy.evaluate_actions(inputs, keep_actions, abstain_actions)
            ratio = torch.exp(out["log_probs"] - old_log_probs)
            clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_batch
            policy_loss = -torch.min(ratio * advantage_batch, clipped).mean()
            value_loss = F.mse_loss(out["values"], return_batch)
            entropy = out["entropy"].mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            losses["policy_loss"].append(float(policy_loss.detach().cpu()))
            losses["value_loss"].append(float(value_loss.detach().cpu()))
            losses["entropy"].append(float(entropy.detach().cpu()))
            losses["total_loss"].append(float(loss.detach().cpu()))

        self.update_count += 1
        return {key: float(np.mean(values)) if values else 0.0 for key, values in losses.items()}

    def train(self) -> Dict[str, Any]:
        start = time.time()
        while self.step_count < self.total_steps:
            rollout_stats = self.collect_rollout()
            if self.buffer.size == 0:
                break
            update_stats = self.update()
            row = {**rollout_stats, **update_stats, "step": float(self.step_count), "update": float(self.update_count)}
            self.history.append(row)

            if self.step_count % self.log_freq < self.rollout_length or self.step_count >= self.total_steps:
                print(
                    "[NQDocPPO] "
                    f"step={self.step_count}/{self.total_steps} "
                    f"reward={rollout_stats['reward']:.3f} "
                    f"acc={rollout_stats['answer_correct']:.3f} "
                    f"asr={rollout_stats['attack_success']:.3f} "
                    f"fpr={rollout_stats['false_positive']:.3f} "
                    f"keep={rollout_stats['kept_ratio']:.3f} "
                    f"qwen_calls={rollout_stats.get('qwen_reward_calls', 0.0):.3f} "
                    f"qwen_delta={rollout_stats.get('qwen_reward_delta', 0.0):.3f} "
                    f"loss={update_stats['total_loss']:.3f}"
                )
            if self.checkpoint_freq > 0 and self.step_count % self.checkpoint_freq < self.rollout_length:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f"nq_doc_policy_step_{self.step_count}.pt"))

        final_path = os.path.join(self.checkpoint_dir, "nq_doc_policy_final.pt")
        self.save_checkpoint(final_path)
        summary = {
            "steps": self.step_count,
            "updates": self.update_count,
            "elapsed_sec": time.time() - start,
            "final_checkpoint": final_path,
            "last": self.history[-1] if self.history else {},
        }
        with open(os.path.join(self.checkpoint_dir, "train_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def save_checkpoint(self, path: str) -> None:
        metadata = {
            "step_count": self.step_count,
            "update_count": self.update_count,
            "trainer_config": self.config,
            "history_tail": self.history[-20:],
        }
        self.policy.save_checkpoint(path, metadata=metadata)
