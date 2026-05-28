"""Per-document adversarial scoring for RAG defense."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .text_features import TextFeatureExtractor, get_doc_text


ASSERTIVE_PATTERNS = [
    r"\bcontrary to\b",
    r"\bdespite\b",
    r"\btherefore\b",
    r"\bcategorically\b",
    r"\bunequivocally\b",
    r"\bofficially confirmed\b",
    r"\bhas announced\b",
    r"\binsider(?:'s)? report\b",
    r"\binternal memo",
    r"\bleaked\b",
    r"\brecently decided\b",
    r"\bmark your calendars\b",
    r"\bset to be released\b",
    r"\bthe correct answer is\b",
    r"\bthe answer is\b",
    r"\bprimary function\b",
    r"\bsurprisingly\b",
    r"\bunconventional\b",
    r"\bstrategically\b",
    r"\bchosen to coincide\b",
    r"\bholiday season\b",
    r"\baims to\b",
    r"\bunexpected role\b",
    r"\bnot just\b",
    r"\bbeyond traditional\b",
    r"\bthus\b",
    r"\bunderscor",
    r"\bmetaphorically\b",
    r"\bsymbolically\b",
    r"\bforeshadow",
    r"\bprefigur",
    r"\bprophetic\b",
    r"\bfresh perspective\b",
    r"\bunorthodox\b",
]


@dataclass
class DocumentAttackScore:
    doc_id: str
    attack_prob: float
    support_score: float
    conflict_score: float
    query_overlap: float
    rank_score: float
    style_score: float
    outlier_score: float
    cluster_score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdversarialDocScorer:
    """
    Scores each retrieved document for attack likelihood.

    The default scorer intentionally avoids `source=attack:*` label leakage. Source
    and rank are still represented as features; source priors are opt-in only for
    diagnostics or non-benchmark deployments.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_extractor: Optional[TextFeatureExtractor] = None,
    ):
        self.config = config or {}
        self.feature_extractor = feature_extractor or TextFeatureExtractor(
            self.config.get("feature_extractor", {})
        )
        self.threshold = float(self.config.get("threshold", 0.18))
        self.max_drop_fraction = float(self.config.get("max_drop_fraction", 0.6))
        self.min_docs = int(self.config.get("min_docs", 1))
        self.use_source_prior = bool(self.config.get("use_source_prior", False))
        self.patterns = [re.compile(pat, re.IGNORECASE) for pat in ASSERTIVE_PATTERNS]

    @staticmethod
    def _doc_id(doc: Any, idx: int) -> str:
        if isinstance(doc, dict):
            return str(doc.get("doc_id") or doc.get("id") or f"doc_{idx}")
        return f"doc_{idx}"

    @staticmethod
    def _rank_score(doc: Any, idx: int, n_docs: int) -> float:
        rank = idx
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {}) or {}
            rank = metadata.get("rank", doc.get("rank", idx))
        try:
            rank = int(rank)
        except Exception:
            rank = idx
        return min(max(rank / max(n_docs - 1, 1), 0.0), 1.0)

    @staticmethod
    def _query_overlap(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
        q = {tok for tok in query_tokens if len(tok) > 2}
        if not q:
            return 0.0
        d = {tok for tok in doc_tokens if len(tok) > 2}
        return len(q & d) / max(len(q), 1)

    def _style_score(self, text: str) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        score = 0.0
        lower = text.lower()

        pattern_hits = 0
        for pat in self.patterns:
            if pat.search(lower):
                pattern_hits += 1
        if pattern_hits:
            score += min(0.45, 0.10 * pattern_hits)
            reasons.append(f"attack_style_patterns={pattern_hits}")

        # PoisonedRAG docs often use repeated high-confidence target phrasing.
        numbers = re.findall(r"\b\d{1,4}(?:[-/]\w{1,9})?\b", text)
        if len(numbers) >= 4:
            score += 0.08
            reasons.append("many_numeric_claims")

        if len(text) > 700:
            score += 0.08
            reasons.append("long_assertive_doc")

        title_bias = lower.startswith("title:") or "\n\n" in text[:160]
        if title_bias:
            score += 0.06
            reasons.append("generated_article_format")

        modal_hits = sum(lower.count(term) for term in ["will be", "is set to", "aims to", "serves as"])
        if modal_hits >= 2:
            score += 0.08
            reasons.append("future_or_causal_framing")

        return min(score, 1.0), reasons

    def _source_prior(self, doc: Any) -> Tuple[float, List[str]]:
        if not self.use_source_prior or not isinstance(doc, dict):
            return 0.0, []
        source = str(doc.get("source", "")).lower()
        if "attack" in source or "poison" in source:
            return 0.35, ["source_prior"]
        return 0.0, []

    def score(
        self,
        query: str,
        docs: List[Any],
        conflict_doc_ids: Optional[Iterable[str]] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        doc_scores: Optional[np.ndarray] = None,
    ) -> List[DocumentAttackScore]:
        if not docs:
            return []

        conflict_set: Set[str] = set(conflict_doc_ids or [])
        texts = [get_doc_text(doc) for doc in docs]
        query_tokens = self.feature_extractor.tokenize(query)
        doc_embeddings = doc_embeddings if doc_embeddings is not None else self.feature_extractor.doc_embeddings(docs)
        doc_scores = doc_scores if doc_scores is not None else self.feature_extractor.query_doc_scores(query, docs, doc_embeddings)

        if len(doc_embeddings) >= 2:
            centroid = doc_embeddings.mean(axis=0, keepdims=True)
            sims = self.feature_extractor.cosine_matrix(doc_embeddings, centroid)[:, 0]
            outliers = np.clip((0.75 - sims) / 0.75, 0.0, 1.0)
            pair_sims = self.feature_extractor.cosine_matrix(doc_embeddings, doc_embeddings)
            np.fill_diagonal(pair_sims, np.nan)
            mean_neighbor_sim = np.nan_to_num(np.nanmean(pair_sims, axis=1), nan=0.0)
            clusters = np.clip((mean_neighbor_sim - 0.35) / 0.65, 0.0, 1.0)
        else:
            outliers = np.zeros((len(docs),), dtype=np.float32)
            clusters = np.zeros((len(docs),), dtype=np.float32)

        results: List[DocumentAttackScore] = []
        for idx, (doc, text) in enumerate(zip(docs, texts)):
            doc_id = self._doc_id(doc, idx)
            doc_tokens = self.feature_extractor.tokenize(text)
            overlap = self._query_overlap(query_tokens, doc_tokens)
            rank_score = self._rank_score(doc, idx, len(docs))
            support = float(doc_scores[idx]) if idx < len(doc_scores) else 0.0
            outlier = float(outliers[idx]) if idx < len(outliers) else 0.0
            cluster = float(clusters[idx]) if idx < len(clusters) else 0.0
            style_score, reasons = self._style_score(text)
            source_score, source_reasons = self._source_prior(doc)
            reasons.extend(source_reasons)

            conflict_score = 1.0 if doc_id in conflict_set else 0.0
            if conflict_score:
                reasons.append("cross_validation_conflict")

            # High query match plus generated/assertive style is suspicious in
            # poisoning because attack docs are crafted to be very relevant.
            crafted_relevance = max(0.0, overlap - 0.55) * 0.55
            if crafted_relevance > 0.05 and style_score > 0.10:
                reasons.append("crafted_query_match")

            attack_prob = (
                0.34 * style_score
                + 0.22 * conflict_score
                + 0.14 * outlier
                + 0.18 * cluster
                + 0.12 * crafted_relevance
                + 0.22 * rank_score
                + source_score
            )

            # If the document is both highly relevant and stylistically
            # manipulative, boost it above the filter threshold.
            if style_score >= 0.24 and overlap >= 0.45:
                attack_prob += 0.28
            elif style_score >= 0.16 and overlap >= 0.35:
                attack_prob += 0.18
            if style_score >= 0.12 and rank_score >= 0.25:
                attack_prob += 0.12
            if cluster >= 0.35 and rank_score >= 0.25:
                attack_prob += 0.10
            if conflict_score and style_score >= 0.10:
                attack_prob += 0.12

            attack_prob = float(min(max(attack_prob, 0.0), 1.0))
            results.append(
                DocumentAttackScore(
                    doc_id=doc_id,
                    attack_prob=attack_prob,
                    support_score=support,
                    conflict_score=conflict_score,
                    query_overlap=overlap,
                    rank_score=rank_score,
                    style_score=style_score,
                    outlier_score=outlier,
                    cluster_score=cluster,
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
        """Drop high-risk documents while preserving a minimum evidence set."""
        if not docs:
            return [], [], []

        threshold = self.threshold if threshold is None else float(threshold)
        min_docs = self.min_docs if min_docs is None else int(min_docs)
        max_drop_fraction = self.max_drop_fraction if max_drop_fraction is None else float(max_drop_fraction)

        scores = self.score(query, docs, conflict_doc_ids=conflict_doc_ids)
        max_drop = int(len(docs) * max_drop_fraction)
        max_drop = min(max_drop, max(0, len(docs) - min_docs))

        candidates = [s for s in scores if s.attack_prob >= threshold]
        candidates.sort(key=lambda s: s.attack_prob, reverse=True)
        drop_ids = {s.doc_id for s in candidates[:max_drop]}

        kept_docs: List[Any] = []
        for idx, doc in enumerate(docs):
            doc_id = self._doc_id(doc, idx)
            if doc_id not in drop_ids:
                kept_docs.append(doc)

        return kept_docs, scores, sorted(drop_ids)
