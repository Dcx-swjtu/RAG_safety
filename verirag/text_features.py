"""Deterministic text features for policy inputs and document scoring."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:  # Optional dependency; the hashed fallback keeps offline runs working.
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency guard
    SentenceTransformer = None


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def get_doc_text(doc: Any) -> str:
    """Return normalized text from either a dict document or a raw string."""
    if isinstance(doc, dict):
        return str(doc.get("text") or doc.get("content") or doc.get("document") or "")
    return str(doc)


class TextFeatureExtractor:
    """
    Build real, deterministic policy features.

    It prefers a local SentenceTransformer/Qwen embedding model when configured,
    and falls back to a hashed bag-of-words encoder. The fallback is not random:
    the same query/document always maps to the same normalized 768-D vector.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.embedding_dim = int(self.config.get("embedding_dim", 768))
        self.max_docs = int(self.config.get("max_docs", 10))
        self.backend = "hashed"
        self.model = None

        model_path = self.config.get("embedding_model_path") or self.config.get("model_path")
        if model_path and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(str(model_path), device=self.config.get("device"))
                self.backend = "sentence_transformer"
                dim = int(self.model.get_sentence_embedding_dimension())
                self.embedding_dim = dim
            except Exception as exc:
                print(f"[TextFeatureExtractor] Falling back to hashed embeddings: {exc}")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts into normalized dense vectors."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self.model is not None:
            emb = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            return emb

        rows = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = self.tokenize(text)
            if not tokens:
                continue
            for pos, tok in enumerate(tokens):
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.embedding_dim
                sign = 1.0 if (digest[4] & 1) else -1.0
                rows[row_idx, bucket] += sign * (1.0 + 0.05 * math.log1p(pos))
            norm = np.linalg.norm(rows[row_idx])
            if norm > 0:
                rows[row_idx] /= norm
        return rows

    @staticmethod
    def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.size == 0 or right.size == 0:
            return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
        left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-8, None)
        right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-8, None)
        return (left_norm @ right_norm.T).astype(np.float32)

    def doc_embeddings(self, docs: List[Any]) -> np.ndarray:
        return self.encode_texts([get_doc_text(doc) for doc in docs])

    def query_doc_scores(self, query: str, docs: List[Any], doc_embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        q_emb = self.encode_texts([query])
        d_emb = doc_embeddings if doc_embeddings is not None else self.doc_embeddings(docs)
        scores = self.cosine_matrix(q_emb, d_emb)[0]
        # Normalize into a stable positive retrieval-like score while preserving rank.
        scores = (scores + 1.0) / 2.0
        return scores.astype(np.float32)

    def build_policy_inputs(
        self,
        query: str,
        docs: List[Any],
        policy_network: Any,
        device: torch.device,
        action_history: Optional[torch.Tensor] = None,
        result_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Create the policy input dict expected by VerificationPolicyNetwork."""
        tokenizer = getattr(policy_network.state_encoder.query_encoder, "tokenizer", None)
        if tokenizer is not None:
            tokens = tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            query_tokens = {k: v.to(device) for k, v in tokens.items()}
        else:
            query_tokens = {
                "input_ids": torch.zeros(1, 512, dtype=torch.long, device=device),
                "attention_mask": torch.ones(1, 512, dtype=torch.long, device=device),
            }

        docs = docs[: self.max_docs]
        doc_emb = self.doc_embeddings(docs)
        doc_scores = self.query_doc_scores(query, docs, doc_emb)

        target_dim = int(getattr(policy_network.state_encoder.doc_encoder, "doc_proj")[0].in_features)
        doc_tensor = torch.zeros(1, len(docs), target_dim, dtype=torch.float32, device=device)
        if len(docs) > 0:
            emb_tensor = torch.from_numpy(doc_emb).to(device=device, dtype=torch.float32)
            if emb_tensor.shape[1] >= target_dim:
                doc_tensor[0] = emb_tensor[:, :target_dim]
            else:
                doc_tensor[0, :, : emb_tensor.shape[1]] = emb_tensor

        score_tensor = torch.from_numpy(doc_scores).to(device=device, dtype=torch.float32).unsqueeze(0)
        doc_masks = torch.ones(1, len(docs), dtype=torch.bool, device=device)

        if action_history is None:
            action_history = torch.zeros(1, 0, dtype=torch.long, device=device)
        if result_history is None:
            result_history = torch.zeros(1, 0, dtype=torch.long, device=device)

        return {
            "query_tokens": query_tokens,
            "query_text": query,
            "doc_embeddings": doc_tensor,
            "doc_scores": score_tensor,
            "doc_masks": doc_masks,
            "action_history": action_history.to(device),
            "result_history": result_history.to(device),
            "history_mask": torch.ones_like(action_history, dtype=torch.bool, device=device),
        }
