"""NQ-focused document-level risk encoder and action policy."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class NQDocRiskEncoder(nn.Module):
    """
    Encode each retrieved document into risk/support signals.

    Inputs are document-level features built by `NQDocumentMaskEnv`, including
    query-doc embedding interactions, retrieval rank, heuristic/learned attack
    score, and verification-style evidence indicators.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        doc_state_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.doc_state_dim = int(doc_state_dim)

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.doc_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.doc_state_dim),
            nn.LayerNorm(self.doc_state_dim),
            nn.GELU(),
        )
        self.signal_head = nn.Linear(self.hidden_dim, 4)

    def forward(self, doc_features: torch.Tensor, doc_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(doc_features)
        doc_state = self.doc_proj(hidden)
        signals = torch.sigmoid(self.signal_head(hidden))
        doc_state = doc_state * doc_mask.unsqueeze(-1).to(doc_state.dtype)
        signals = signals * doc_mask.unsqueeze(-1).to(signals.dtype)
        return {
            "doc_state": doc_state,
            "attack_prob": signals[..., 0],
            "support_prob": signals[..., 1],
            "conflict_prob": signals[..., 2],
            "relevance_prob": signals[..., 3],
        }


class NQDocumentActionPolicy(nn.Module):
    """
    Per-document keep/drop policy with a global abstain action.

    Action:
    - `keep_mask`: [B, max_docs] binary keep decisions.
    - `abstain`: [B] binary global abstain decision.
    """

    def __init__(
        self,
        input_dim: int = 3088,
        hidden_dim: int = 256,
        doc_state_dim: int = 128,
        global_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = {
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "doc_state_dim": int(doc_state_dim),
            "global_dim": int(global_dim),
            "dropout": float(dropout),
        }
        self.risk_encoder = NQDocRiskEncoder(input_dim, hidden_dim, doc_state_dim, dropout)
        self.keep_head = nn.Sequential(
            nn.Linear(doc_state_dim + 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(doc_state_dim * 2 + 8, global_dim),
            nn.LayerNorm(global_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.abstain_head = nn.Linear(global_dim, 1)
        self.value_head = nn.Linear(global_dim, 1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        doc_features = inputs["doc_features"]
        doc_mask = inputs["doc_mask"].bool()
        encoder_out = self.risk_encoder(doc_features, doc_mask)
        doc_state = encoder_out["doc_state"]
        signal_stack = torch.stack(
            [
                encoder_out["attack_prob"],
                encoder_out["support_prob"],
                encoder_out["conflict_prob"],
                encoder_out["relevance_prob"],
            ],
            dim=-1,
        )

        keep_logits = self.keep_head(torch.cat([doc_state, signal_stack], dim=-1)).squeeze(-1)
        keep_logits = keep_logits.masked_fill(~doc_mask, -20.0)

        mask_float = doc_mask.to(doc_state.dtype)
        denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_state = (doc_state * mask_float.unsqueeze(-1)).sum(dim=1) / denom
        max_state = doc_state.masked_fill(~doc_mask.unsqueeze(-1), -1e4).max(dim=1).values
        max_state = torch.where(torch.isfinite(max_state), max_state, torch.zeros_like(max_state))

        signal_mean = (signal_stack * mask_float.unsqueeze(-1)).sum(dim=1) / denom
        signal_max = signal_stack.masked_fill(~doc_mask.unsqueeze(-1), -1e4).max(dim=1).values
        signal_max = torch.where(torch.isfinite(signal_max), signal_max, torch.zeros_like(signal_max))

        global_state = self.global_proj(torch.cat([mean_state, max_state, signal_mean, signal_max], dim=-1))
        abstain_logits = self.abstain_head(global_state).squeeze(-1)
        value = self.value_head(global_state).squeeze(-1)
        keep_probs = torch.sigmoid(keep_logits) * mask_float
        abstain_prob = torch.sigmoid(abstain_logits)
        return {
            **encoder_out,
            "keep_logits": keep_logits,
            "keep_probs": keep_probs,
            "abstain_logits": abstain_logits,
            "abstain_prob": abstain_prob,
            "value": value,
            "global_state": global_state,
        }

    def select_action(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, Any]:
        with torch.no_grad():
            out = self.forward(inputs)
        doc_mask = inputs["doc_mask"].bool()
        keep_probs = out["keep_probs"].clamp(1e-5, 1.0 - 1e-5)
        abstain_prob = out["abstain_prob"].clamp(1e-5, 1.0 - 1e-5)

        if deterministic:
            keep_action = (keep_probs >= 0.5).float()
            abstain_action = (abstain_prob >= 0.5).float()
        else:
            keep_action = Bernoulli(probs=keep_probs).sample()
            abstain_action = Bernoulli(probs=abstain_prob).sample()

        keep_action = keep_action * doc_mask.to(keep_action.dtype)
        for row in range(keep_action.size(0)):
            if abstain_action[row] < 0.5 and keep_action[row].sum() < 0.5 and doc_mask[row].any():
                best_idx = torch.argmax(keep_probs[row].masked_fill(~doc_mask[row], -1.0))
                keep_action[row, best_idx] = 1.0

        keep_dist = Bernoulli(probs=keep_probs)
        abstain_dist = Bernoulli(probs=abstain_prob)
        keep_log_prob = keep_dist.log_prob(keep_action).masked_fill(~doc_mask, 0.0).sum(dim=1)
        abstain_log_prob = abstain_dist.log_prob(abstain_action).view(-1)
        log_prob = keep_log_prob + abstain_log_prob

        return {
            "keep_mask": keep_action,
            "abstain": abstain_action,
            "log_prob": log_prob,
            "value": out["value"],
            "keep_probs": keep_probs,
            "abstain_prob": abstain_prob,
            "attack_prob": out["attack_prob"],
            "support_prob": out["support_prob"],
            "conflict_prob": out["conflict_prob"],
        }

    def evaluate_actions(
        self,
        inputs: Dict[str, torch.Tensor],
        keep_actions: torch.Tensor,
        abstain_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(inputs)
        doc_mask = inputs["doc_mask"].bool()
        keep_probs = out["keep_probs"].clamp(1e-5, 1.0 - 1e-5)
        abstain_prob = out["abstain_prob"].clamp(1e-5, 1.0 - 1e-5)

        keep_dist = Bernoulli(probs=keep_probs)
        abstain_dist = Bernoulli(probs=abstain_prob)
        keep_log_probs = keep_dist.log_prob(keep_actions).masked_fill(~doc_mask, 0.0).sum(dim=1)
        abstain_log_probs = abstain_dist.log_prob(abstain_actions.view(-1)).view(-1)
        keep_entropy = keep_dist.entropy().masked_fill(~doc_mask, 0.0).sum(dim=1)
        abstain_entropy = abstain_dist.entropy().view(-1)
        return {
            "log_probs": keep_log_probs + abstain_log_probs,
            "values": out["value"],
            "entropy": keep_entropy + abstain_entropy,
            "keep_probs": keep_probs,
            "abstain_prob": abstain_prob,
        }

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        torch.save(
            {
                "model_state": self.state_dict(),
                "state_dict": self.state_dict(),
                "config": self.config,
                "metadata": metadata or {},
            },
            path,
        )

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state", checkpoint.get("state_dict"))
        if state_dict is None:
            raise KeyError("Checkpoint must contain model_state or state_dict")
        self.load_state_dict(state_dict, strict=strict)
        return checkpoint.get("metadata", {})
