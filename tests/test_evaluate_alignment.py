import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import VeriRAGEvaluator


class TestEvaluateAlignment(unittest.TestCase):
    def test_check_answer_supports_aliases_and_containment(self):
        self.assertTrue(
            VeriRAGEvaluator._check_answer("The date is February 2.", ["Feb. 2", "February 2"])
        )

    def test_check_answer_rejects_target_answer(self):
        self.assertFalse(VeriRAGEvaluator._check_answer("March 15", ["February 2"]))

    def test_gold_sample_excludes_weak_document_labels(self):
        weak = {"metadata": {"answer_source": "weak_document_label"}}
        gold = {"metadata": {"answer_source": "poisonedrag_targeted_results", "eval_gold": True}}
        self.assertFalse(VeriRAGEvaluator._is_gold_sample(weak))
        self.assertTrue(VeriRAGEvaluator._is_gold_sample(gold))

    def test_load_test_data_respects_split_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            row = {"id": "ms_marco_validation_1", "query": "q", "answers": ["a"], "documents": []}
            (data_dir / "ms_marco_validation.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            evaluator = VeriRAGEvaluator.__new__(VeriRAGEvaluator)
            evaluator.config = {"data": {"data_dir": str(data_dir)}, "evaluation": {"split_map": {"ms_marco": "validation"}}}
            evaluator.n_questions = 1
            loaded = evaluator._load_test_data("ms_marco")
            self.assertEqual(loaded[0]["id"], "ms_marco_validation_1")


    def test_load_attack_data_indexes_sample_id_and_query_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            attacks_dir = data_dir / "attacks"
            attacks_dir.mkdir()
            row = {
                "sample_id": "sample-1",
                "query_id": "query-1",
                "poisoned_documents": ["poison text"],
            }
            (attacks_dir / "nq_poisonedrag.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            evaluator = VeriRAGEvaluator.__new__(VeriRAGEvaluator)
            evaluator.config = {"data": {"data_dir": str(data_dir)}}
            loaded = evaluator._load_attack_data("nq", "poisonedrag")
            self.assertEqual(loaded["sample-1"], row)
            self.assertEqual(loaded["query-1"], row)

    def test_extract_poisoned_texts_supports_string_and_dict_docs(self):
        docs = [
            "plain text",
            {"text": "text field"},
            {"document": "document field"},
            {"content": "content field"},
            {"text": ""},
            123,
        ]
        self.assertEqual(
            VeriRAGEvaluator._extract_poisoned_texts(docs),
            ["plain text", "text field", "document field", "content field"],
        )

    def test_require_fixed_attacks_raises_when_attack_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = VeriRAGEvaluator.__new__(VeriRAGEvaluator)
            evaluator.config = {"data": {"data_dir": tmp}}
            evaluator.require_fixed_attacks = True
            with self.assertRaises(FileNotFoundError):
                evaluator._evaluate_attacks(
                    [{"id": "sample-1", "query": "q", "target_answer": "bad", "documents": [], "metadata": {"query_id": "query-1"}}],
                    "nq",
                    "poisonedrag",
                )

    def test_configured_attack_types_override_default(self):
        evaluator = VeriRAGEvaluator(
            policy_network=None,
            defense_orchestrator=None,
            attack_simulator=None,
            config={"evaluation": {"attack_types": ["poisonedrag"]}},
        )
        self.assertEqual(evaluator.attack_types, ["poisonedrag"])



if __name__ == "__main__":
    unittest.main()
