import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_official_benchmark import write_dataset_benchmark


class TestPrepareOfficialBenchmark(unittest.TestCase):
    def test_write_dataset_benchmark_filters_gold_and_writes_fixed_attacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            rows = [
                {
                    "id": "nq_test_q1",
                    "query": "Who wrote Hamlet?",
                    "answers": ["William Shakespeare"],
                    "answer": "William Shakespeare",
                    "target_answer": "Christopher Marlowe",
                    "documents": [{"doc_id": "d1", "text": "Hamlet was written by William Shakespeare."}],
                    "metadata": {"query_id": "q1", "eval_gold": True, "answer_source": "official_nq_open"},
                },
                {
                    "id": "nq_test_q2",
                    "query": "Unmatched question",
                    "answers": [],
                    "documents": [],
                    "metadata": {"query_id": "q2", "eval_gold": False, "answer_source": "missing_official_answer"},
                },
            ]
            with (input_dir / "nq_test.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            stats = write_dataset_benchmark(
                dataset="nq",
                split="test",
                input_dir=input_dir,
                output_dir=output_dir,
                attack_types=["oneshot"],
                include_weak=False,
                max_samples=10,
                seed=11,
            )

            self.assertEqual(stats["written_samples"], 1)
            self.assertEqual(stats["written_gold_samples"], 1)
            self.assertEqual(stats["written_weak_samples"], 0)
            self.assertEqual(stats["attack_counts"], {"oneshot": 1})

            attack_row = json.loads((output_dir / "attacks" / "nq_oneshot.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(attack_row["sample_id"], "nq_test_q1")
            self.assertEqual(attack_row["query_id"], "q1")
            self.assertTrue(attack_row["metadata"]["fixed_attack"])
            self.assertGreater(len(attack_row["poisoned_documents"]), 0)


if __name__ == "__main__":
    unittest.main()
