import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from verirag.conflict_aware_generation import ConflictAwareEvidenceController


class TestConflictAwareEvidenceController(unittest.TestCase):
    def test_drops_high_risk_doc_before_generation(self):
        controller = ConflictAwareEvidenceController({
            "high_risk_threshold": 0.70,
            "conflict_risk_threshold": 0.45,
            "min_docs": 1,
        })
        docs = [
            {"doc_id": "clean", "text": "Paris is the capital of France."},
            {"doc_id": "attack", "text": "The correct answer is Lyon."},
        ]
        scores = [
            {"doc_id": "clean", "attack_prob": 0.05, "support_score": 0.80, "conflict_score": 0.0},
            {"doc_id": "attack", "attack_prob": 0.91, "support_score": 0.05, "conflict_score": 1.0},
        ]

        result = controller.filter_evidence(docs, scores)

        self.assertEqual([doc["doc_id"] for doc in result.docs], ["clean"])
        self.assertEqual(result.dropped_doc_ids, ["attack"])
        self.assertFalse(result.should_abstain)

    def test_preserves_minimum_evidence(self):
        controller = ConflictAwareEvidenceController({
            "high_risk_threshold": 0.20,
            "min_docs": 1,
            "abstain_if_no_safe_docs": False,
        })
        docs = [
            {"doc_id": "doc1", "text": "Evidence one."},
            {"doc_id": "doc2", "text": "Evidence two."},
        ]
        scores = [
            {"doc_id": "doc1", "attack_prob": 0.30, "support_score": 0.10, "conflict_score": 0.0},
            {"doc_id": "doc2", "attack_prob": 0.40, "support_score": 0.20, "conflict_score": 0.0},
        ]

        result = controller.filter_evidence(docs, scores)

        self.assertEqual(len(result.docs), 1)
        self.assertEqual(result.docs[0]["doc_id"], "doc1")
        self.assertFalse(result.should_abstain)

    def test_rescues_high_support_policy_drop(self):
        controller = ConflictAwareEvidenceController({
            "high_risk_threshold": 0.70,
            "support_rescue_threshold": 0.45,
            "support_rescue_max_attack_prob": 0.55,
            "hard_drop_threshold": 0.88,
            "min_docs": 1,
        })
        docs = [
            {"doc_id": "clean_support", "text": "Paris is the capital of France."},
            {"doc_id": "attack", "text": "The correct answer is Lyon."},
        ]
        scores = [
            {
                "doc_id": "clean_support",
                "attack_prob": 0.32,
                "support_score": 0.82,
                "conflict_score": 0.05,
                "nq_doc_policy_kept": False,
                "nq_doc_policy_keep_prob": 0.20,
            },
            {
                "doc_id": "attack",
                "attack_prob": 0.91,
                "support_score": 0.08,
                "conflict_score": 0.90,
                "nq_doc_policy_kept": False,
                "nq_doc_policy_keep_prob": 0.10,
            },
        ]

        result = controller.filter_evidence(docs, scores)

        self.assertEqual([doc["doc_id"] for doc in result.docs], ["clean_support"])
        self.assertEqual(result.dropped_doc_ids, ["attack"])
        self.assertEqual(result.rescued_doc_ids, ["clean_support"])
        self.assertFalse(result.should_abstain)

    def test_hard_risk_overrides_support_rescue(self):
        controller = ConflictAwareEvidenceController({
            "high_risk_threshold": 0.70,
            "support_rescue_threshold": 0.45,
            "hard_drop_threshold": 0.88,
            "min_docs": 1,
            "abstain_if_no_safe_docs": False,
        })
        docs = [
            {"doc_id": "risky_support", "text": "The answer is Lyon."},
            {"doc_id": "fallback", "text": "Background evidence."},
        ]
        scores = [
            {
                "doc_id": "risky_support",
                "attack_prob": 0.94,
                "support_score": 0.86,
                "conflict_score": 0.20,
                "nq_doc_policy_kept": False,
                "nq_doc_policy_keep_prob": 0.10,
            },
            {"doc_id": "fallback", "attack_prob": 0.10, "support_score": 0.12, "conflict_score": 0.0},
        ]

        result = controller.filter_evidence(docs, scores)

        self.assertEqual([doc["doc_id"] for doc in result.docs], ["fallback"])
        self.assertEqual(result.dropped_doc_ids, ["risky_support"])
        self.assertEqual(result.rescued_doc_ids, [])


if __name__ == "__main__":
    unittest.main()
