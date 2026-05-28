"""
Import official QA answers into BEIR-style VeriRAG evaluation files.

BEIR supplies query/corpus/qrels. Official QA files supply answers. Samples that
cannot be matched to an official answer are still written, but marked
``eval_gold=false`` so scripts/evaluate.py excludes them from clean ACC.

Examples:
  python scripts/import_official_answers.py \
    --beir-data /path/to/beir \
    --nq-official /path/to/nq-open.dev.jsonl \
    --msmarco-official /path/to/msmarco-dev.json \
    --hotpotqa-official /path/to/hotpot_dev_fullwiki_v1.json \
    --output ./data_official_aligned
"""

import argparse
import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


DATASET_DIR_ALIASES = {
    "ms_marco": "msmarco",
}

DATASET_ANSWER_SOURCES = {
    "nq": "official_nq_open",
    "hotpotqa": "official_hotpotqa",
    "ms_marco": "official_msmarco",
}

SPLIT_ALIASES = {
    "train": "train",
    "dev": "validation",
    "validation": "validation",
    "test": "test",
}

ID_KEYS = ("id", "_id", "qid", "query_id", "query-id", "queryId")
QUESTION_KEYS = ("question", "query", "text", "question_text")
ANSWER_KEYS = (
    "answers",
    "answer",
    "short_answers",
    "short_answer",
    "wellFormedAnswers",
    "well_formed_answers",
    "correct answer",
)
NO_ANSWER_VALUES = {
    "",
    "[]",
    "none",
    "null",
    "no answer",
    "no answer present.",
    "noanswer",
    "n/a",
}


@dataclass
class AnswerMatch:
    answers: List[str]
    source: str
    raw_id: Optional[str] = None
    question: Optional[str] = None
    match_method: str = "unknown"


class AnswerIndex:
    def __init__(self) -> None:
        self.by_id: Dict[str, AnswerMatch] = {}
        self.by_question: Dict[str, AnswerMatch] = {}

    def add(
        self,
        answers: Sequence[str],
        source: str,
        raw_id: Optional[Any] = None,
        question: Optional[str] = None,
    ) -> None:
        clean_answers = dedupe([answer for answer in normalize_answers(answers) if answer])
        if not clean_answers:
            return
        match = AnswerMatch(
            answers=clean_answers,
            source=source,
            raw_id=str(raw_id) if raw_id is not None else None,
            question=question,
        )
        if raw_id is not None and str(raw_id).strip():
            self.by_id[str(raw_id).strip()] = match
        if question:
            key = normalize_question(question)
            if key:
                self.by_question[key] = match

    def lookup(self, query_id: str, question: str) -> Optional[AnswerMatch]:
        match = self.by_id.get(str(query_id))
        if match is not None:
            return AnswerMatch(
                answers=match.answers,
                source=match.source,
                raw_id=match.raw_id,
                question=match.question,
                match_method="id",
            )
        match = self.by_question.get(normalize_question(question))
        if match is not None:
            return AnswerMatch(
                answers=match.answers,
                source=match.source,
                raw_id=match.raw_id,
                question=match.question,
                match_method="question_text",
            )
        return None

    def __len__(self) -> int:
        return len(self.by_id) + len(self.by_question)


def normalize_question(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def dedupe(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        key = value.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def normalize_answers(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ANSWER_KEYS + ("text", "value", "alias"):
            if key in value:
                answers = normalize_answers(value[key])
                if answers:
                    return answers
        return []
    if isinstance(value, (list, tuple, set)):
        answers: List[str] = []
        for item in value:
            answers.extend(normalize_answers(item))
        return dedupe(answers)
    text = str(value).strip()
    if text.casefold() in NO_ANSWER_VALUES:
        return []
    return [text]


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> Any:
    with open_text(path) as f:
        return json.load(f)


def extract_id(row: Dict[str, Any]) -> Optional[str]:
    for key in ID_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def extract_question(row: Dict[str, Any]) -> Optional[str]:
    for key in QUESTION_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def extract_answers(row: Dict[str, Any]) -> List[str]:
    for key in ANSWER_KEYS:
        if key in row:
            answers = normalize_answers(row[key])
            if answers:
                return answers
    return []


def iter_parquet_rows(path: Path) -> Iterable[Dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            yield row


def iter_table_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with open_text(path) as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames and len(reader.fieldnames) > 1:
            yield from reader
            return
        f.seek(0)
        simple_reader = csv.reader(f, delimiter=delimiter)
        for parts in simple_reader:
            if len(parts) >= 2:
                yield {"id": parts[0], "answer": parts[1:] if len(parts) > 2 else parts[1]}


def add_generic_row(index: AnswerIndex, row: Dict[str, Any], source: str, fallback_id: Optional[str] = None) -> None:
    raw_id = extract_id(row) or fallback_id
    question = extract_question(row)
    answers = extract_answers(row)
    index.add(answers=answers, source=source, raw_id=raw_id, question=question)


def add_generic_object(index: AnswerIndex, data: Any, source: str) -> None:
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                add_generic_row(index, row, source)
        return

    if not isinstance(data, dict):
        return

    for container_key in ("data", "examples", "questions", "items"):
        if isinstance(data.get(container_key), list):
            add_generic_object(index, data[container_key], source)
            return

    if "answers" in data and isinstance(data.get("answers"), dict):
        # MS MARCO QA style: {"query": {qid: text}, "answers": {qid: [...]}}
        query_map = data.get("query") or data.get("queries") or {}
        answers_map = data.get("answers") or {}
        well_formed_map = data.get("wellFormedAnswers") or data.get("well_formed_answers") or {}
        query_ids = set(answers_map) | set(well_formed_map)
        for query_id in query_ids:
            answers = normalize_answers(answers_map.get(query_id))
            answers.extend(normalize_answers(well_formed_map.get(query_id)))
            question = query_map.get(query_id) if isinstance(query_map, dict) else None
            index.add(answers=answers, source=source, raw_id=query_id, question=question)
        return

    if extract_answers(data):
        add_generic_row(index, data, source)
        return

    # Common id -> record or id -> answer map.
    for key, value in data.items():
        if isinstance(value, dict):
            row = dict(value)
            for id_key in ("id", "_id", "qid", "query_id"):
                row.setdefault(id_key, key)
            add_generic_row(index, row, source, fallback_id=str(key))
        else:
            index.add(answers=normalize_answers(value), source=source, raw_id=str(key))


def load_official_answers(dataset_name: str, path: Optional[Path]) -> AnswerIndex:
    index = AnswerIndex()
    if path is None:
        return index
    if not path.exists():
        raise FileNotFoundError(f"Official answer file not found: {path}")

    source = DATASET_ANSWER_SOURCES[dataset_name]
    suffixes = path.suffixes
    is_jsonl = ".jsonl" in suffixes or path.name.endswith(".jsonl.gz")
    is_json = ".json" in suffixes or path.name.endswith(".json.gz")
    is_table = path.suffix in {".tsv", ".csv"} or path.name.endswith((".tsv.gz", ".csv.gz"))
    is_parquet = path.suffix == ".parquet"

    if is_jsonl:
        for row in read_jsonl(path):
            add_generic_row(index, row, source)
        return index
    if is_json:
        add_generic_object(index, load_json(path), source)
        return index
    if is_table:
        for row in iter_table_rows(path):
            add_generic_row(index, row, source)
        return index
    if is_parquet:
        for row in iter_parquet_rows(path):
            add_generic_row(index, row, source)
        return index

    raise ValueError(f"Unsupported official answer format: {path}")


def load_queries(path: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("_id", row.get("id"))): row
        for row in read_jsonl(path)
        if row.get("_id", row.get("id")) is not None
    }


def read_qrels(path: Path) -> Dict[str, List[str]]:
    qrels: Dict[str, List[str]] = {}
    if not path.exists():
        return qrels
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if first and parts[0] in {"query-id", "query_id", "qid"}:
                first = False
                continue
            first = False
            if len(parts) < 2:
                continue
            query_id, doc_id = parts[0], parts[1]
            qrels.setdefault(query_id, [])
            if doc_id not in qrels[query_id]:
                qrels[query_id].append(doc_id)
    return qrels


def load_split_qrels(qrels_dir: Path, requested_splits: Set[str]) -> Dict[str, Dict[str, List[str]]]:
    split_qrels: Dict[str, Dict[str, List[str]]] = {}
    for path in sorted(qrels_dir.glob("*.tsv")):
        raw_split = path.stem
        output_split = SPLIT_ALIASES.get(raw_split, raw_split)
        if raw_split not in requested_splits and output_split not in requested_splits:
            continue
        qrels = read_qrels(path)
        if qrels:
            split_qrels[output_split] = qrels
    return split_qrels


def collect_needed_doc_ids(split_qrels: Dict[str, Dict[str, List[str]]], max_docs: int) -> Set[str]:
    needed: Set[str] = set()
    for qrels in split_qrels.values():
        for doc_ids in qrels.values():
            needed.update(doc_ids[:max_docs])
    return needed


def load_needed_corpus(path: Path, needed_doc_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    docs: Dict[str, Dict[str, Any]] = {}
    remaining = set(needed_doc_ids)
    for row in read_jsonl(path):
        doc_id = str(row.get("_id", row.get("id", "")))
        if doc_id in remaining:
            docs[doc_id] = row
            remaining.remove(doc_id)
            if not remaining:
                break
    return docs


def doc_to_runtime(dataset_name: str, doc_id: str, doc: Dict[str, Any], rank: int) -> Dict[str, Any]:
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    full_text = f"{title}\n{text}".strip() if title else text
    return {
        "doc_id": doc_id,
        "title": title,
        "text": full_text,
        "source": dataset_name,
        "metadata": {
            "rank": rank,
            "selection_source": "qrels",
            **(doc.get("metadata", {}) or {}),
        },
    }


def build_sample(
    dataset_name: str,
    split_name: str,
    query_id: str,
    query: Dict[str, Any],
    doc_ids: List[str],
    corpus: Dict[str, Dict[str, Any]],
    answer_match: Optional[AnswerMatch],
) -> Optional[Dict[str, Any]]:
    docs = []
    for rank, doc_id in enumerate(doc_ids):
        doc = corpus.get(doc_id)
        if doc is not None:
            docs.append(doc_to_runtime(dataset_name, doc_id, doc, rank))
    if not docs:
        return None

    question = str(query.get("text", query.get("question", ""))).strip()
    answers = answer_match.answers if answer_match else []
    joined_document = "\n".join(doc["text"] for doc in docs)
    answer_source = answer_match.source if answer_match else "missing_official_answer"
    eval_gold = bool(answers)
    target_answer = f"NOT {answers[0]}" if answers else "unsupported answer"

    return {
        "id": f"{dataset_name}_{split_name}_{query_id}",
        "dataset": dataset_name,
        "question": question,
        "query": question,
        "answers": answers,
        "answer": answers[0] if answers else "",
        "ground_truth": answers[0] if answers else "",
        "target_answer": target_answer,
        "documents": docs,
        "document": joined_document,
        "text": joined_document,
        "metadata": {
            "dataset": dataset_name,
            "split": split_name,
            "source_format": "beir_with_official_answers",
            "query_id": query_id,
            "qrel_doc_ids": doc_ids,
            "selected_doc_ids": [doc["doc_id"] for doc in docs],
            "answer_source": answer_source,
            "eval_gold": eval_gold,
            "answer_match_method": answer_match.match_method if answer_match else None,
            "official_answer_id": answer_match.raw_id if answer_match else None,
        },
    }


def convert_dataset(
    dataset_name: str,
    source_root: Path,
    output_dir: Path,
    answer_index: AnswerIndex,
    max_docs: int,
    requested_splits: Set[str],
) -> Dict[str, Any]:
    source_name = DATASET_DIR_ALIASES.get(dataset_name, dataset_name)
    dataset_dir = source_root / source_name
    queries_path = dataset_dir / "queries.jsonl"
    corpus_path = dataset_dir / "corpus.jsonl"
    qrels_dir = dataset_dir / "qrels"
    if not queries_path.exists() or not corpus_path.exists() or not qrels_dir.exists():
        raise FileNotFoundError(f"Missing BEIR files under {dataset_dir}")

    queries = load_queries(queries_path)
    split_qrels = load_split_qrels(qrels_dir, requested_splits)
    needed_doc_ids = collect_needed_doc_ids(split_qrels, max_docs=max_docs)
    print(f"[OfficialAlign] {dataset_name}: loading {len(needed_doc_ids)} docs")
    corpus = load_needed_corpus(corpus_path, needed_doc_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "dataset": dataset_name,
        "official_answer_entries": len(answer_index),
        "splits": {},
    }

    for split_name, qrels in split_qrels.items():
        output_path = output_dir / f"{dataset_name}_{split_name}.jsonl"
        unmatched_path = output_dir / f"{dataset_name}_{split_name}_unmatched_answers.jsonl"
        written = 0
        gold = 0
        weak = 0
        skipped_no_docs = 0
        match_methods: Dict[str, int] = {}
        with output_path.open("w", encoding="utf-8") as out_f, unmatched_path.open("w", encoding="utf-8") as unmatched_f:
            for query_id, qrel_doc_ids in qrels.items():
                query = queries.get(query_id)
                if not query:
                    continue
                question = str(query.get("text", query.get("question", ""))).strip()
                answer_match = answer_index.lookup(query_id, question)
                sample = build_sample(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    query_id=query_id,
                    query=query,
                    doc_ids=qrel_doc_ids[:max_docs],
                    corpus=corpus,
                    answer_match=answer_match,
                )
                if sample is None:
                    skipped_no_docs += 1
                    continue
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
                if sample["metadata"].get("eval_gold"):
                    gold += 1
                    method = sample["metadata"].get("answer_match_method") or "unknown"
                    match_methods[method] = match_methods.get(method, 0) + 1
                else:
                    weak += 1
                    unmatched_f.write(json.dumps({"query_id": query_id, "question": question}, ensure_ascii=False) + "\n")

        if weak == 0:
            unmatched_path.unlink(missing_ok=True)
        manifest["splits"][split_name] = {
            "qrel_queries": len(qrels),
            "written_samples": written,
            "gold_answer_samples": gold,
            "weak_unmatched_samples": weak,
            "skipped_no_docs": skipped_no_docs,
            "coverage": gold / max(written, 1),
            "match_methods": match_methods,
            "output": str(output_path),
            "unmatched_output": str(unmatched_path) if weak else None,
        }
        print(
            f"[OfficialAlign] {dataset_name}/{split_name}: "
            f"written={written} gold={gold} weak={weak} coverage={gold / max(written, 1):.4f} -> {output_path}"
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official QA answers into BEIR-style VeriRAG data")
    parser.add_argument("--beir-data", required=True, help="BEIR data root with dataset/query/corpus/qrels files")
    parser.add_argument("--output", default="./data_official_aligned", help="Output directory")
    parser.add_argument("--nq-official", default=None, help="NQ-open official answers JSON/JSONL/TSV")
    parser.add_argument("--msmarco-official", default=None, help="MS MARCO QA official answers JSON/JSONL/TSV")
    parser.add_argument("--hotpotqa-official", default=None, help="HotpotQA official answers JSON/JSONL/TSV")
    parser.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "ms_marco"])
    parser.add_argument("--splits", nargs="+", default=["test", "dev", "validation"])
    parser.add_argument("--max-docs", type=int, default=5)
    args = parser.parse_args()

    official_paths = {
        "nq": Path(args.nq_official) if args.nq_official else None,
        "hotpotqa": Path(args.hotpotqa_official) if args.hotpotqa_official else None,
        "ms_marco": Path(args.msmarco_official) if args.msmarco_official else None,
    }
    requested_splits = {SPLIT_ALIASES.get(split, split) for split in args.splits} | set(args.splits)
    output_dir = Path(args.output)
    manifest: Dict[str, Any] = {
        "beir_data": str(Path(args.beir_data).resolve()),
        "output": str(output_dir.resolve()),
        "max_docs": args.max_docs,
        "requested_splits": sorted(requested_splits),
        "official_paths": {key: str(value.resolve()) if value else None for key, value in official_paths.items()},
        "datasets": {},
    }

    for dataset_name in args.datasets:
        answer_index = load_official_answers(dataset_name, official_paths.get(dataset_name))
        manifest["datasets"][dataset_name] = convert_dataset(
            dataset_name=dataset_name,
            source_root=Path(args.beir_data),
            output_dir=output_dir,
            answer_index=answer_index,
            max_docs=args.max_docs,
            requested_splits=requested_splits,
        )

    manifest_path = output_dir / "official_alignment_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[OfficialAlign] Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
