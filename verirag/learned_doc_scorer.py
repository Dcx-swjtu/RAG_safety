"""Learned per-document adversarial scorer for RAG defense."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .adversarial_doc_scorer import AdversarialDocScorer, DocumentAttackScore
from .text_features import TextFeatureExtractor, get_doc_text


class AdversarialDocClassifier(nn.Module):
    """Small MLP over query-doc semantic and retrieval/risk features."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class DocTrainingExample:
    dataset: str
    sample_id: str
    query: str
    doc: Dict[str, Any]
    label: int
    attack_type: str = "clean"


def _doc_id(doc: Any, idx: int) -> str:
    if isinstance(doc, dict):
        return str(doc.get("doc_id") or doc.get("id") or f"doc_{idx}")
    return f"doc_{idx}"


def _normalize_docs(docs: Any) -> List[Dict[str, Any]]:
    if not isinstance(docs, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        if isinstance(doc, dict):
            normalized = dict(doc)
            normalized.setdefault("doc_id", normalized.get("id", f"doc_{idx}"))
            normalized.setdefault("text", get_doc_text(normalized))
            normalized.setdefault("metadata", normalized.get("metadata", {}) or {})
            normalized["metadata"].setdefault("rank", idx)
            normalized.setdefault("source", normalized.get("source", "retrieved"))
            out.append(normalized)
        else:
            out.append(
                {
                    "doc_id": f"doc_{idx}",
                    "text": str(doc),
                    "source": "retrieved",
                    "metadata": {"rank": idx},
                }
            )
    return out


def _attack_texts(row: Dict[str, Any]) -> List[str]:
    poisoned = row.get("poisoned_documents", [])
    if not isinstance(poisoned, list):
        return []
    texts: List[str] = []
    for doc in poisoned:
        if isinstance(doc, dict):
            text = doc.get("text") or doc.get("content") or doc.get("document") or ""
        else:
            text = str(doc)
        text = str(text).strip()
        if text:
            texts.append(text)
    return texts


def build_doc_classifier_examples(
    data_dir: str,
    datasets: Sequence[str],
    attack_types: Sequence[str],
    max_clean_docs_per_sample: int = 10,
    max_attack_docs_per_sample: int = 10,
    splits: Sequence[str] = ("train", "validation", "test"),
) -> List[DocTrainingExample]:
    """Create clean/poisoned document labels from aligned benchmark files."""
    root = Path(data_dir)
    examples: List[DocTrainingExample] = []

    for dataset in datasets:
        rows: List[Dict[str, Any]] = []
        active_split = ""
        for split in splits:
            path = root / f"{dataset}_{split}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if rows:
                active_split = split
                break

        samples_by_key: Dict[str, Dict[str, Any]] = {}
        for sample in rows:
            metadata = sample.get("metadata", {}) or {}
            for key in (sample.get("id"), metadata.get("query_id")):
                if key:
                    samples_by_key[str(key)] = sample

        for sample in rows:
            query = sample.get("query") or sample.get("question") or ""
            sample_id = str(sample.get("id") or sample.get("metadata", {}).get("query_id") or "")
            for idx, doc in enumerate(_normalize_docs(sample.get("documents", []))[:max_clean_docs_per_sample]):
                clean_doc = dict(doc)
                clean_doc.setdefault("metadata", {})
                clean_doc["metadata"] = dict(clean_doc["metadata"])
                clean_doc["metadata"].setdefault("rank", idx)
                examples.append(DocTrainingExample(dataset, sample_id, query, clean_doc, 0, "clean"))

        for attack_type in attack_types:
            path = root / "attacks" / active_split / f"{dataset}_{attack_type}.jsonl"
            if not path.exists():
                path = root / "attacks" / f"{dataset}_{attack_type}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sample = (
                        samples_by_key.get(str(row.get("sample_id")))
                        or samples_by_key.get(str(row.get("query_id")))
                    )
                    if sample is None:
                        continue
                    query = row.get("query") or sample.get("query") or sample.get("question") or ""
                    sample_id = str(row.get("sample_id") or row.get("query_id") or "")
                    for idx, text in enumerate(_attack_texts(row)[:max_attack_docs_per_sample]):
                        examples.append(
                            DocTrainingExample(
                                dataset=dataset,
                                sample_id=sample_id,
                                query=query,
                                doc={
                                    "doc_id": f"attack_{attack_type}_{idx}",
                                    "text": text,
                                    "source": "poisoned",
                                    "metadata": {"rank": max_clean_docs_per_sample + idx},
                                },
                                label=1,
                                attack_type=attack_type,
                            )
                        )

    return examples


class LearnedAdversarialDocScorer:
    """
    Learned scorer with heuristic fallback/ensemble.

    The model consumes deterministic query/doc embeddings plus explicit numeric
    features. It never uses `source=attack:*` as an input feature.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_extractor: Optional[TextFeatureExtractor] = None,
        heuristic_scorer: Optional[AdversarialDocScorer] = None,
    ):
        self.config = config or {}
        self.feature_extractor = feature_extractor or TextFeatureExtractor(
            self.config.get("feature_extractor", {})
        )
        self.heuristic_scorer = heuristic_scorer or AdversarialDocScorer(
            self.config.get("heuristic", {}),
            feature_extractor=self.feature_extractor,
        )
        self.threshold = float(self.config.get("threshold", 0.18))
        self.max_drop_fraction = float(self.config.get("max_drop_fraction", 0.85))
        self.min_docs = int(self.config.get("min_docs", 1))
        self.hidden_dim = int(self.config.get("hidden_dim", 512))
        self.dropout = float(self.config.get("dropout", 0.1))
        self.ensemble_weight = float(self.config.get("ensemble_weight", 1.0))
        self.device = torch.device(self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.input_dim = int(self.config.get("input_dim", self.feature_extractor.embedding_dim * 4 + 8))
        self.model = AdversarialDocClassifier(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        self.model.eval()
        self.loaded = False

        model_path = self.config.get("model_path") or self.config.get("checkpoint")
        if model_path:
            self.load(model_path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        cfg = checkpoint.get("config", {})
        input_dim = int(cfg.get("input_dim", checkpoint.get("input_dim", self.input_dim)))
        hidden_dim = int(cfg.get("hidden_dim", self.hidden_dim))
        dropout = float(cfg.get("dropout", self.dropout))
        if input_dim != self.input_dim or hidden_dim != self.hidden_dim:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.model = AdversarialDocClassifier(input_dim, hidden_dim, dropout).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.loaded = True

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "dropout": self.dropout,
                    "embedding_dim": self.feature_extractor.embedding_dim,
                },
                "metadata": metadata or {},
            },
            path,
        )

    def build_features(
        self,
        query: str,
        docs: List[Any],
        heuristic_scores: Optional[List[DocumentAttackScore]] = None,
    ) -> np.ndarray:
        if not docs:
            return np.zeros((0, self.input_dim), dtype=np.float32)
        q_emb = self.feature_extractor.encode_texts([query])
        d_emb = self.feature_extractor.doc_embeddings(docs)
        q_rep = np.repeat(q_emb, len(docs), axis=0)
        query_doc_scores = self.feature_extractor.query_doc_scores(query, docs, d_emb)
        heuristic_scores = heuristic_scores if heuristic_scores is not None else self.heuristic_scorer.score(query, docs)

        rows: List[np.ndarray] = []
        query_tokens = self.feature_extractor.tokenize(query)
        for idx, (doc, h_score) in enumerate(zip(docs, heuristic_scores)):
            doc_tokens = self.feature_extractor.tokenize(get_doc_text(doc))
            overlap = self.heuristic_scorer._query_overlap(query_tokens, doc_tokens)
            rank_score = self.heuristic_scorer._rank_score(doc, idx, len(docs))
            text_len = min(len(get_doc_text(doc)) / 2000.0, 1.0)
            token_len = min(len(doc_tokens) / 400.0, 1.0)
            numeric = np.asarray(
                [
                    query_doc_scores[idx],
                    rank_score,
                    overlap,
                    h_score.style_score,
                    h_score.outlier_score,
                    h_score.cluster_score,
                    h_score.conflict_score,
                    text_len + 0.1 * token_len,
                ],
                dtype=np.float32,
            )
            row = np.concatenate(
                [
                    q_rep[idx],
                    d_emb[idx],
                    np.abs(q_rep[idx] - d_emb[idx]),
                    q_rep[idx] * d_emb[idx],
                    numeric,
                ],
                axis=0,
            ).astype(np.float32)
            if row.shape[0] < self.input_dim:
                row = np.pad(row, (0, self.input_dim - row.shape[0]))
            elif row.shape[0] > self.input_dim:
                row = row[: self.input_dim]
            rows.append(row)
        return np.stack(rows, axis=0)

    def score(
        self,
        query: str,
        docs: List[Any],
        conflict_doc_ids: Optional[Iterable[str]] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        doc_scores: Optional[np.ndarray] = None,
    ) -> List[DocumentAttackScore]:
        heuristic_scores = self.heuristic_scorer.score(
            query,
            docs,
            conflict_doc_ids=conflict_doc_ids,
            doc_embeddings=doc_embeddings,
            doc_scores=doc_scores,
        )
        if not docs or not self.loaded:
            return heuristic_scores

        features = self.build_features(query, docs, heuristic_scores=heuristic_scores)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(features).to(self.device))
            learned_probs = torch.sigmoid(logits).cpu().numpy()

        results: List[DocumentAttackScore] = []
        for idx, base in enumerate(heuristic_scores):
            learned_prob = float(learned_probs[idx])
            attack_prob = (
                self.ensemble_weight * learned_prob
                + (1.0 - self.ensemble_weight) * base.attack_prob
            )
            attack_prob = float(min(max(attack_prob, 0.0), 1.0))
            reasons = list(base.reasons)
            reasons.append(f"learned_attack_prob={learned_prob:.3f}")
            results.append(
                DocumentAttackScore(
                    doc_id=base.doc_id,
                    attack_prob=attack_prob,
                    support_score=base.support_score,
                    conflict_score=base.conflict_score,
                    query_overlap=base.query_overlap,
                    rank_score=base.rank_score,
                    style_score=base.style_score,
                    outlier_score=base.outlier_score,
                    cluster_score=base.cluster_score,
                    reasons=reasons,
                )
            )
        return results

    def filter_docs(
        self,
        query: str,
        docs: List[Any],
        conflict_doc_ids: Optional[Iterable[str]] = None,
        threshold: Optional[float] = None,
        min_docs: Optional[int] = None,
        max_drop_fraction: Optional[float] = None,
    ) -> Tuple[List[Any], List[DocumentAttackScore], List[str]]:
        if not docs:
            return [], [], []
        threshold = self.threshold if threshold is None else float(threshold)
        min_docs = self.min_docs if min_docs is None else int(min_docs)
        max_drop_fraction = self.max_drop_fraction if max_drop_fraction is None else float(max_drop_fraction)
        scores = self.score(query, docs, conflict_doc_ids=conflict_doc_ids)
        max_drop = int(math.floor(len(docs) * max_drop_fraction))
        max_drop = min(max_drop, max(0, len(docs) - min_docs))
        candidates = [score for score in scores if score.attack_prob >= threshold]
        candidates.sort(key=lambda score: score.attack_prob, reverse=True)
        drop_ids = {score.doc_id for score in candidates[:max_drop]}
        kept_docs = [
            doc for idx, doc in enumerate(docs)
            if _doc_id(doc, idx) not in drop_ids
        ]
        return kept_docs, scores, sorted(drop_ids)
