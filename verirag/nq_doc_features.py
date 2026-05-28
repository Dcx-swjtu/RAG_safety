"""Reusable NQ document-level policy feature builder."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .adversarial_doc_scorer import AdversarialDocScorer
from .text_features import TextFeatureExtractor, get_doc_text


def _digit_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    query_digits = {tok for tok in query_tokens if tok.isdigit()}
    doc_digits = {tok for tok in doc_tokens if tok.isdigit()}
    if not query_digits:
        return 0.0
    return len(query_digits & doc_digits) / max(len(query_digits), 1)


def _title_overlap(query_tokens: List[str], title: str, extractor: TextFeatureExtractor) -> float:
    title_tokens = extractor.tokenize(title)
    if not query_tokens or not title_tokens:
        return 0.0
    return len(set(query_tokens) & set(title_tokens)) / max(len(set(query_tokens)), 1)


class NQDocFeatureBuilder:
    """Build policy features available both during training and deployment."""

    def __init__(self, feature_extractor: TextFeatureExtractor, doc_scorer: Any):
        self.feature_extractor = feature_extractor
        self.doc_scorer = doc_scorer

    @property
    def feature_dim(self) -> int:
        return self.feature_extractor.embedding_dim * 4 + 16

    def build(self, query: str, docs: List[Dict[str, Any]]) -> np.ndarray:
        if not docs:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        q_emb = self.feature_extractor.encode_texts([query])
        d_emb = self.feature_extractor.doc_embeddings(docs)
        q_rep = np.repeat(q_emb, len(docs), axis=0)
        doc_scores = self.doc_scorer.score(query, docs)
        query_scores = self.feature_extractor.query_doc_scores(query, docs, d_emb)
        doc_doc_sim = self.feature_extractor.cosine_matrix(d_emb, d_emb)
        query_tokens = self.feature_extractor.tokenize(query)
        rows: List[np.ndarray] = []

        for idx, (doc, score) in enumerate(zip(docs, doc_scores)):
            text = get_doc_text(doc)
            doc_tokens = self.feature_extractor.tokenize(text)
            if len(docs) > 1:
                neighbors = np.delete(doc_doc_sim[idx], idx)
                doc_doc_mean = float(np.mean((neighbors + 1.0) / 2.0))
                doc_doc_max = float(np.max((neighbors + 1.0) / 2.0))
            else:
                doc_doc_mean = 0.0
                doc_doc_max = 0.0
            rank_score = AdversarialDocScorer._rank_score(doc, idx, len(docs))
            query_overlap = AdversarialDocScorer._query_overlap(query_tokens, doc_tokens)
            normalized_rank = idx / max(len(docs) - 1, 1)
            unique_ratio = len(set(doc_tokens)) / max(len(doc_tokens), 1)
            title = str(doc.get("title", "")) if isinstance(doc, dict) else ""
            numeric = np.asarray(
                [
                    query_scores[idx],
                    rank_score,
                    query_overlap,
                    score.attack_prob,
                    score.support_score,
                    score.conflict_score,
                    score.style_score,
                    score.outlier_score,
                    score.cluster_score,
                    doc_doc_mean,
                    doc_doc_max,
                    normalized_rank,
                    min(len(text) / 2000.0, 1.0),
                    min(len(doc_tokens) / 400.0, 1.0),
                    _digit_overlap(query_tokens, doc_tokens),
                    _title_overlap(query_tokens, title, self.feature_extractor),
                ],
                dtype=np.float32,
            )
            row = np.concatenate(
                [q_rep[idx], d_emb[idx], np.abs(q_rep[idx] - d_emb[idx]), q_rep[idx] * d_emb[idx], numeric],
                axis=0,
            ).astype(np.float32)
            rows.append(row)
        return np.stack(rows, axis=0)
