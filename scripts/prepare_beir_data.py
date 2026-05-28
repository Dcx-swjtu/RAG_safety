"""
Convert BEIR-style retrieval datasets into VeriRAG runtime JSONL splits.

Expected source layout:
  <source>/<dataset>/queries.jsonl
  <source>/<dataset>/corpus.jsonl
  <source>/<dataset>/qrels/{train,dev,test}.tsv

The output files match scripts/train.py and scripts/evaluate.py:
  <output>/<dataset>_{train,validation,test}.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DATASET_DIR_ALIASES = {
    "ms_marco": "msmarco",
}

SPLIT_ALIASES = {
    "train": "train",
    "dev": "validation",
    "validation": "validation",
    "test": "test",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_queries(path: Path) -> Dict[str, Dict[str, Any]]:
    queries = {}
    for row in read_jsonl(path):
        query_id = str(row.get("_id", row.get("id", "")))
        if query_id:
            queries[query_id] = row
    return queries


def read_qrels(path: Path, max_queries: int) -> Dict[str, List[str]]:
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
            if query_id not in qrels and len(qrels) >= max_queries:
                continue
            qrels.setdefault(query_id, [])
            if doc_id not in qrels[query_id]:
                qrels[query_id].append(doc_id)
    return qrels


def collect_needed_doc_ids(split_qrels: Dict[str, Dict[str, List[str]]]) -> Set[str]:
    needed: Set[str] = set()
    for qrels in split_qrels.values():
        for doc_ids in qrels.values():
            needed.update(doc_ids)
    return needed


def load_needed_corpus(path: Path, needed_doc_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    docs = {}
    remaining = set(needed_doc_ids)
    for row in read_jsonl(path):
        doc_id = str(row.get("_id", row.get("id", "")))
        if doc_id in remaining:
            docs[doc_id] = row
            remaining.remove(doc_id)
            if not remaining:
                break
    return docs


def first_sentence(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    for marker in [". ", "? ", "! "]:
        idx = text.find(marker)
        if 0 < idx < max_chars:
            return text[: idx + 1]
    return text[:max_chars]


def weak_answer(query: Dict[str, Any], docs: List[Dict[str, Any]]) -> str:
    metadata = query.get("metadata", {}) or {}
    answer = metadata.get("answer")
    if answer:
        return str(answer)
    for doc in docs:
        title = str(doc.get("title", "")).strip()
        if title:
            return title
    for doc in docs:
        sentence = first_sentence(str(doc.get("text", "")))
        if sentence:
            return sentence
    return ""


def to_runtime_sample(
    dataset_name: str,
    split_name: str,
    query_id: str,
    query: Dict[str, Any],
    doc_ids: List[str],
    corpus: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    docs = []
    for rank, doc_id in enumerate(doc_ids):
        doc = corpus.get(doc_id)
        if not doc:
            continue
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        full_text = f"{title}\n{text}".strip() if title else text
        docs.append(
            {
                "doc_id": doc_id,
                "text": full_text,
                "source": dataset_name,
                "metadata": {
                    "rank": rank,
                    **(doc.get("metadata", {}) or {}),
                },
            }
        )
    if not docs:
        return None

    question = str(query.get("text", query.get("question", ""))).strip()
    answer = weak_answer(query, [corpus[doc_id] for doc_id in doc_ids if doc_id in corpus])
    joined_document = "\n".join(doc["text"] for doc in docs)
    metadata = {
        "dataset": dataset_name,
        "split": split_name,
        "source_format": "beir",
        "query_id": query_id,
        "qrel_doc_ids": doc_ids,
        "answer_source": "query_metadata" if (query.get("metadata", {}) or {}).get("answer") else "weak_document_label",
        **(query.get("metadata", {}) or {}),
    }
    return {
        "id": f"{dataset_name}_{split_name}_{query_id}",
        "question": question,
        "query": question,
        "answer": answer,
        "ground_truth": answer,
        "target_answer": f"NOT {answer}" if answer else "unsupported answer",
        "documents": docs,
        "document": joined_document,
        "text": joined_document,
        "metadata": metadata,
    }


def convert_dataset(
    dataset_name: str,
    source_root: Path,
    output_dir: Path,
    max_queries_per_split: int,
) -> Dict[str, int]:
    source_name = DATASET_DIR_ALIASES.get(dataset_name, dataset_name)
    dataset_dir = source_root / source_name
    queries_path = dataset_dir / "queries.jsonl"
    corpus_path = dataset_dir / "corpus.jsonl"
    qrels_dir = dataset_dir / "qrels"

    if not queries_path.exists() or not corpus_path.exists() or not qrels_dir.exists():
        raise FileNotFoundError(f"Missing BEIR files under {dataset_dir}")

    print(f"[BEIR] Loading queries: {queries_path}")
    queries = load_queries(queries_path)

    split_qrels: Dict[str, Dict[str, List[str]]] = {}
    for qrel_name, output_split in SPLIT_ALIASES.items():
        qrels = read_qrels(qrels_dir / f"{qrel_name}.tsv", max_queries=max_queries_per_split)
        if qrels and output_split not in split_qrels:
            split_qrels[output_split] = qrels

    if "train" not in split_qrels and "test" in split_qrels:
        split_qrels["train"] = dict(list(split_qrels["test"].items())[:max_queries_per_split])
    if "validation" not in split_qrels and "test" in split_qrels:
        validation_items = list(split_qrels["test"].items())[: max(1, max_queries_per_split // 5)]
        split_qrels["validation"] = dict(validation_items)

    needed_doc_ids = collect_needed_doc_ids(split_qrels)
    print(f"[BEIR] Loading {len(needed_doc_ids)} referenced docs from {corpus_path}")
    corpus = load_needed_corpus(corpus_path, needed_doc_ids)
    print(f"[BEIR] Loaded docs: {len(corpus)}/{len(needed_doc_ids)}")

    counts = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        qrels = split_qrels.get(split_name, {})
        output_path = output_dir / f"{dataset_name}_{split_name}.jsonl"
        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for query_id, doc_ids in qrels.items():
                query = queries.get(query_id)
                if not query:
                    continue
                sample = to_runtime_sample(dataset_name, split_name, query_id, query, doc_ids, corpus)
                if sample is None:
                    continue
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
        counts[f"{dataset_name}_{split_name}"] = count
        print(f"[BEIR] {dataset_name}/{split_name}: {count} -> {output_path}")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Prepare BEIR-style data for VeriRAG")
    parser.add_argument("--source", required=True, help="Source BEIR data root")
    parser.add_argument("--output", required=True, help="Output data directory")
    parser.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "ms_marco"])
    parser.add_argument("--max-queries-per-split", type=int, default=1000)
    args = parser.parse_args()

    source_root = Path(args.source)
    output_dir = Path(args.output)
    all_counts: Dict[str, int] = {}
    for dataset_name in args.datasets:
        counts = convert_dataset(dataset_name, source_root, output_dir, args.max_queries_per_split)
        all_counts.update(counts)

    stats_path = output_dir / "data_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(all_counts, f, indent=2, ensure_ascii=False)
    print(f"[BEIR] Statistics -> {stats_path}")


if __name__ == "__main__":
    main()
