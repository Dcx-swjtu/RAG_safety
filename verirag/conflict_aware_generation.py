"""Conflict-aware evidence control before RAG answer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from .adversarial_doc_scorer import AdversarialDocScorer
from .cross_validator import ConsistencyLabel, ValidationReport


@dataclass
class ConflictAwareResult:
    """Evidence decision used by the final answer generation stage."""

    docs: List[Dict[str, Any]]
    dropped_doc_ids: List[str] = field(default_factory=list)
    should_abstain: bool = False
    abstain_reason: str = ""
    risk_score: float = 0.0
    notes: List[str] = field(default_factory=list)


class ConflictAwareEvidenceController:
    """
    Final evidence controller for document-level RAG defense.

    The scorer/policy stage removes obviously adversarial documents. This stage
    handles residual conflicts immediately before generation so the generator is
    only exposed to low-risk, conflict-free evidence.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.high_risk_threshold = float(self.config.get("high_risk_threshold", 0.70))
        self.conflict_risk_threshold = float(self.config.get("conflict_risk_threshold", 0.45))
        self.support_floor = float(self.config.get("support_floor", 0.10))
        self.min_docs = int(self.config.get("min_docs", 1))
        self.max_drop_fraction = float(self.config.get("max_drop_fraction", 0.85))
        self.abstain_if_no_safe_docs = bool(self.config.get("abstain_if_no_safe_docs", True))

    def filter_evidence(
        self,
        docs: List[Dict[str, Any]],
        doc_scores: Optional[Sequence[Dict[str, Any]]] = None,
        validation_reports: Optional[Sequence[ValidationReport]] = None,
    ) -> ConflictAwareResult:
        if not docs:
            return ConflictAwareResult(
                docs=[],
                should_abstain=self.abstain_if_no_safe_docs,
                abstain_reason="no_evidence",
                risk_score=1.0,
                notes=["no_docs_available"],
            )

        score_by_id = self._score_by_id(doc_scores or [])
        conflict_doc_ids = self._conflict_doc_ids(validation_reports or [])
        risky_ids = self._candidate_drop_ids(docs, score_by_id, conflict_doc_ids)
        max_drop = min(
            int(len(docs) * self.max_drop_fraction),
            max(0, len(docs) - self.min_docs),
        )

        ranked_drop_ids = sorted(
            risky_ids,
            key=lambda doc_id: self._risk_value(score_by_id.get(doc_id, {}), doc_id in conflict_doc_ids),
            reverse=True,
        )
        drop_ids = set(ranked_drop_ids[:max_drop])
        kept_docs = [
            doc for idx, doc in enumerate(docs)
            if self._doc_id(doc, idx) not in drop_ids
        ]

        notes: List[str] = []
        if conflict_doc_ids:
            notes.append(f"conflict_docs={len(conflict_doc_ids)}")
        if drop_ids:
            notes.append(f"conflict_aware_drop={len(drop_ids)}")

        if not kept_docs:
            if self.abstain_if_no_safe_docs:
                return ConflictAwareResult(
                    docs=[],
                    dropped_doc_ids=sorted(drop_ids),
                    should_abstain=True,
                    abstain_reason="no_safe_evidence_after_conflict_filter",
                    risk_score=1.0,
                    notes=notes,
                )
            best_doc = self._lowest_risk_doc(docs, score_by_id)
            kept_docs = [best_doc]
            best_id = self._doc_id(best_doc, docs.index(best_doc))
            drop_ids.discard(best_id)
            notes.append("retained_lowest_risk_doc")

        residual_risk = self._residual_risk(kept_docs, score_by_id, conflict_doc_ids)
        should_abstain = bool(
            residual_risk >= self.high_risk_threshold
            and self._max_support(kept_docs, score_by_id) < self.support_floor
        )
        return ConflictAwareResult(
            docs=kept_docs,
            dropped_doc_ids=sorted(drop_ids),
            should_abstain=should_abstain,
            abstain_reason="residual_conflict_risk" if should_abstain else "",
            risk_score=residual_risk,
            notes=notes,
        )

    @staticmethod
    def generation_instruction(query: str) -> str:
        return (
            "Use only the remaining conflict-free evidence. If the evidence is "
            "insufficient or mutually inconsistent, say that the answer cannot "
            f"be determined. Question: {query}"
        )

    @staticmethod
    def _doc_id(doc: Dict[str, Any], idx: int) -> str:
        return str(doc.get("doc_id") or doc.get("id") or f"doc_{idx}")

    @staticmethod
    def _score_by_id(doc_scores: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in doc_scores:
            doc_id = row.get("doc_id")
            if doc_id is not None:
                out[str(doc_id)] = dict(row)
        return out

    @staticmethod
    def _conflict_doc_ids(reports: Sequence[ValidationReport]) -> Set[str]:
        conflict_doc_ids: Set[str] = set()
        for report in reports:
            if report.label != ConsistencyLabel.CONFLICT:
                continue
            for claim in getattr(report, "source_claims", []) or []:
                doc_id = getattr(claim, "doc_id", None)
                if doc_id:
                    conflict_doc_ids.add(str(doc_id))
        return conflict_doc_ids

    def _candidate_drop_ids(
        self,
        docs: List[Dict[str, Any]],
        score_by_id: Dict[str, Dict[str, Any]],
        conflict_doc_ids: Iterable[str],
    ) -> Set[str]:
        conflict_set = set(conflict_doc_ids)
        out: Set[str] = set()
        for idx, doc in enumerate(docs):
            doc_id = self._doc_id(doc, idx)
            row = score_by_id.get(doc_id, {})
            attack_prob = float(row.get("attack_prob", 0.0))
            conflict_score = float(row.get("conflict_score", 0.0))
            if attack_prob >= self.high_risk_threshold:
                out.add(doc_id)
            elif doc_id in conflict_set and attack_prob >= self.conflict_risk_threshold:
                out.add(doc_id)
            elif conflict_score > 0.0 and attack_prob >= self.conflict_risk_threshold:
                out.add(doc_id)
        return out

    def _risk_value(self, row: Dict[str, Any], is_conflict: bool) -> float:
        return (
            float(row.get("attack_prob", 0.0))
            + 0.30 * float(row.get("conflict_score", 0.0))
            + (0.20 if is_conflict else 0.0)
            - 0.15 * float(row.get("support_score", 0.0))
        )

    def _residual_risk(
        self,
        docs: List[Dict[str, Any]],
        score_by_id: Dict[str, Dict[str, Any]],
        conflict_doc_ids: Iterable[str],
    ) -> float:
        if not docs:
            return 1.0
        conflict_set = set(conflict_doc_ids)
        risks = []
        for idx, doc in enumerate(docs):
            doc_id = self._doc_id(doc, idx)
            risks.append(self._risk_value(score_by_id.get(doc_id, {}), doc_id in conflict_set))
        return float(np.clip(max(risks), 0.0, 1.0))

    def _max_support(
        self,
        docs: List[Dict[str, Any]],
        score_by_id: Dict[str, Dict[str, Any]],
    ) -> float:
        if not docs:
            return 0.0
        supports = []
        for idx, doc in enumerate(docs):
            doc_id = self._doc_id(doc, idx)
            row = score_by_id.get(doc_id, {})
            supports.append(float(row.get("support_score", 0.0)))
        return max(supports, default=0.0)

    def _lowest_risk_doc(
        self,
        docs: List[Dict[str, Any]],
        score_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        best_idx = 0
        best_risk = float("inf")
        for idx, doc in enumerate(docs):
            doc_id = self._doc_id(doc, idx)
            risk = self._risk_value(score_by_id.get(doc_id, {}), False)
            rank = AdversarialDocScorer._rank_score(doc, idx, len(docs))
            keyed = risk + 0.01 * rank
            if keyed < best_risk:
                best_idx = idx
                best_risk = keyed
        return docs[best_idx]
