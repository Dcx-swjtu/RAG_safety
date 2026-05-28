import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.import_official_answers import convert_dataset, load_official_answers


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestImportOfficialAnswers(unittest.TestCase):
    def test_nq_open_question_text_alignment_marks_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "beir"
            dataset = root / "nq"
            write_jsonl(dataset / "queries.jsonl", [{"_id": "test1", "text": "What is the capital of France?"}])
            write_jsonl(dataset / "corpus.jsonl", [{"_id": "doc1", "title": "France", "text": "Paris is the capital."}])
            (dataset / "qrels").mkdir(parents=True)
            (dataset / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\ntest1\tdoc1\t1\n")

            official = Path(tmp) / "nq-open.jsonl"
            write_jsonl(official, [{"question": "what is the capital of france", "answer": ["Paris"]}])
            index = load_official_answers("nq", official)
            manifest = convert_dataset("nq", root, Path(tmp) / "out", index, max_docs=5, requested_splits={"test"})

            row = json.loads(((Path(tmp) / "out" / "nq_test.jsonl").read_text()).splitlines()[0])
            self.assertEqual(row["answers"], ["Paris"])
            self.assertTrue(row["metadata"]["eval_gold"])
            self.assertEqual(row["metadata"]["answer_match_method"], "question_text")
            self.assertEqual(manifest["splits"]["test"]["gold_answer_samples"], 1)

    def test_msmarco_dict_answers_align_by_query_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            official = Path(tmp) / "msmarco.json"
            official.write_text(
                json.dumps(
                    {
                        "query": {"1163399": "what day is groundhog's day?"},
                        "answers": {"1163399": ["February 2"]},
                        "wellFormedAnswers": {"1163399": ["Groundhog Day is February 2."]},
                    }
                ),
                encoding="utf-8",
            )
            index = load_official_answers("ms_marco", official)
            match = index.lookup("1163399", "different text")
            self.assertIsNotNone(match)
            self.assertIn("February 2", match.answers)
            self.assertEqual(match.match_method, "id")

    def test_msmarco_parquet_answers_align_by_query_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            import pyarrow as pa
            import pyarrow.parquet as pq

            official = Path(tmp) / "msmarco.parquet"
            table = pa.table(
                {
                    "query_id": [1163399],
                    "query": ["what day is groundhog's day?"],
                    "answers": [["February 2"]],
                    "wellFormedAnswers": [["Groundhog Day is February 2."]],
                }
            )
            pq.write_table(table, official)
            index = load_official_answers("ms_marco", official)
            match = index.lookup("1163399", "different text")
            self.assertIsNotNone(match)
            self.assertIn("February 2", match.answers)
            self.assertEqual(match.match_method, "id")

    def test_unmatched_queries_are_written_as_non_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "beir"
            dataset = root / "hotpotqa"
            write_jsonl(dataset / "queries.jsonl", [{"_id": "h1", "text": "Unknown question?"}])
            write_jsonl(dataset / "corpus.jsonl", [{"_id": "doc1", "title": "Doc", "text": "No answer here."}])
            (dataset / "qrels").mkdir(parents=True)
            (dataset / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nh1\tdoc1\t1\n")

            manifest = convert_dataset("hotpotqa", root, Path(tmp) / "out", load_official_answers("hotpotqa", None), 5, {"test"})
            row = json.loads(((Path(tmp) / "out" / "hotpotqa_test.jsonl").read_text()).splitlines()[0])
            self.assertFalse(row["metadata"]["eval_gold"])
            self.assertEqual(row["answers"], [])
            self.assertEqual(manifest["splits"]["test"]["weak_unmatched_samples"], 1)
            self.assertTrue((Path(tmp) / "out" / "hotpotqa_test_unmatched_answers.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
