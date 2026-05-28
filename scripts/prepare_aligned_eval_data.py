"""
Prepare fully aligned VeriRAG evaluation data.

This script aligns three pieces of data into one auditable benchmark:
- BEIR corpus/qrels for source documents.
- PoisonedRAG targeted results for gold answers, target answers, and
  published adversarial documents.
- VeriRAG AttackSimulator outputs for the non-PoisonedRAG attack types.

Output:
  <output>/<dataset>_test.jsonl
  <output>/attacks/<dataset>_<attack_type>.jsonl
  <output>/alignment_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from verirag.attack_simulator import AttackSimulator


DATASET_DIR_ALIASES = {
    "ms_marco": "msmarco",
}

BEIR_RESULT_NAMES = {
    "nq": "nq-contriever.json",
    "hotpotqa": "hotpotqa-contriever.json",
    "ms_marco": "msmarco-contriever.json",
}

TARGETED_RESULT_NAMES = {
    "nq": "nq.json",
    "hotpotqa": "hotpotqa.json",
    "ms_marco": "msmarco.json",
}

ATTACK_TYPES = [
    "poisonedrag",
    "oneshot",
    "refinerag",
    "semantic_chameleon",
    "adaptive",
]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_queries(path: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("_id", row.get("id"))): row
        for row in read_jsonl(path)
        if row.get("_id", row.get("id")) is not None
    }


def load_qrels(qrels_dir: Path) -> Dict[str, List[str]]:
    qrels: Dict[str, List[str]] = {}
    for path in sorted(qrels_dir.glob("*.tsv")):
        with path.open("r", encoding="utf-8") as f:
            first = True
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                if first and parts[0] in {"query-id", "query_id", "qid"}:
                    first = False
                    continue
                first = False
                query_id, doc_id = parts[0], parts[1]
                qrels.setdefault(query_id, [])
                if doc_id not in qrels[query_id]:
                    qrels[query_id].append(doc_id)
    return qrels


def load_retrieval(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    retrieval = {}
    for query_id, scored_docs in raw.items():
        retrieval[query_id] = list(scored_docs.keys())
    return retrieval


def collect_doc_ids(doc_ids_by_query: Dict[str, List[str]]) -> Set[str]:
    needed: Set[str] = set()
    for doc_ids in doc_ids_by_query.values():
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


def select_doc_ids(query_id: str, qrels: Dict[str, List[str]], retrieval: Dict[str, List[str]], max_docs: int) -> List[str]:
    selected: List[str] = []
    for doc_id in qrels.get(query_id, []):
        if doc_id not in selected:
            selected.append(doc_id)
        if len(selected) >= max_docs:
            return selected
    for doc_id in retrieval.get(query_id, []):
        if doc_id not in selected:
            selected.append(doc_id)
        if len(selected) >= max_docs:
            break
    return selected


def normalize_answer(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def doc_to_runtime(dataset_name: str, doc_id: str, doc: Dict[str, Any], rank: int, source: str) -> Dict[str, Any]:
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    full_text = f"{title}\n{text}".strip() if title else text
    return {
        "doc_id": doc_id,
        "title": title,
        "text": full_text,
        "source": dataset_name,
        "score": 1.0 if source == "qrels" else None,
        "metadata": {
            "rank": rank,
            "selection_source": source,
            **(doc.get("metadata", {}) or {}),
        },
    }


def build_sample(
    dataset_name: str,
    query_id: str,
    targeted: Dict[str, Any],
    query_row: Dict[str, Any],
    doc_ids: List[str],
    corpus: Dict[str, Dict[str, Any]],
    qrels: Dict[str, List[str]],
) -> Dict[str, Any]:
    docs = []
    for rank, doc_id in enumerate(doc_ids):
        if doc_id not in corpus:
            continue
        source = "qrels" if doc_id in qrels.get(query_id, []) else "retrieval"
        docs.append(doc_to_runtime(dataset_name, doc_id, corpus[doc_id], rank, source))

    answers = normalize_answer(targeted.get("correct answer"))
    target_answer = str(targeted.get("incorrect answer", "")).strip()
    question = str(targeted.get("question") or query_row.get("text") or "").strip()
    joined_document = "\n".join(doc["text"] for doc in docs)

    return {
        "id": f"{dataset_name}_aligned_{query_id}",
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
            "split": "test",
            "query_id": query_id,
            "source_format": "aligned_beir_poisonedrag",
            "answer_source": "poisonedrag_targeted_results",
            "target_source": "poisonedrag_targeted_results",
            "doc_source": "qrels_plus_contriever_retrieval",
            "eval_gold": bool(answers),
            "qrel_doc_ids": qrels.get(query_id, []),
            "selected_doc_ids": doc_ids,
            "adv_text_count": len(targeted.get("adv_texts", [])),
        },
    }


def write_attacks(
    dataset_name: str,
    samples: List[Dict[str, Any]],
    targeted_rows: Dict[str, Dict[str, Any]],
    output_dir: Path,
    attack_simulator: AttackSimulator,
) -> Dict[str, int]:
    attacks_dir = output_dir / "attacks"
    attacks_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for attack_type in ATTACK_TYPES:
        output_path = attacks_dir / f"{dataset_name}_{attack_type}.jsonl"
        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                query_id = sample["metadata"]["query_id"]
                targeted = targeted_rows[query_id]
                original_docs = [doc.get("text", "") for doc in sample.get("documents", [])]
                if attack_type == "poisonedrag":
                    poisoned_documents = list(targeted.get("adv_texts", []))
                else:
                    poisoned_documents = attack_simulator.generate(
                        query=sample["query"],
                        target_answer=sample["target_answer"],
                        attack_type=attack_type,
                        original_docs=original_docs,
                    )
                row = {
                    "id": f"{sample['id']}_{attack_type}",
                    "sample_id": sample["id"],
                    "query_id": query_id,
                    "dataset": dataset_name,
                    "attack_type": attack_type,
                    "query": sample["query"],
                    "answers": sample.get("answers", []),
                    "target_answer": sample["target_answer"],
                    "poisoned_documents": poisoned_documents,
                    "metadata": {
                        "attack_source": "poisonedrag_adv_texts" if attack_type == "poisonedrag" else "verirag_attack_simulator",
                        "fixed_attack": True,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        counts[attack_type] = count
        print(f"[Aligned] {dataset_name}/{attack_type}: {count} -> {output_path}")
    return counts


def convert_dataset(
    dataset_name: str,
    source_root: Path,
    targeted_root: Path,
    retrieval_root: Path,
    output_dir: Path,
    max_docs: int,
    attack_simulator: AttackSimulator,
) -> Dict[str, Any]:
    source_name = DATASET_DIR_ALIASES.get(dataset_name, dataset_name)
    dataset_dir = source_root / source_name
    targeted_path = targeted_root / TARGETED_RESULT_NAMES[dataset_name]
    retrieval_path = retrieval_root / BEIR_RESULT_NAMES[dataset_name]

    targeted_rows: Dict[str, Dict[str, Any]] = json.loads(targeted_path.read_text())
    queries = load_queries(dataset_dir / "queries.jsonl")
    qrels = load_qrels(dataset_dir / "qrels")
    retrieval = load_retrieval(retrieval_path)

    doc_ids_by_query = {
        query_id: select_doc_ids(query_id, qrels, retrieval, max_docs)
        for query_id in targeted_rows
    }
    needed_doc_ids = collect_doc_ids(doc_ids_by_query)
    print(f"[Aligned] {dataset_name}: loading {len(needed_doc_ids)} docs")
    corpus = load_needed_corpus(dataset_dir / "corpus.jsonl", needed_doc_ids)

    samples = []
    skipped = []
    qrels_used = 0
    retrieval_used = 0
    for query_id, targeted in targeted_rows.items():
        query_row = queries.get(query_id, {})
        doc_ids = [doc_id for doc_id in doc_ids_by_query.get(query_id, []) if doc_id in corpus]
        if not doc_ids:
            skipped.append(query_id)
            continue
        if any(doc_id in qrels.get(query_id, []) for doc_id in doc_ids):
            qrels_used += 1
        else:
            retrieval_used += 1
        samples.append(build_sample(dataset_name, query_id, targeted, query_row, doc_ids, corpus, qrels))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_test.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[Aligned] {dataset_name}/test: {len(samples)} -> {output_path}")

    attack_counts = write_attacks(dataset_name, samples, targeted_rows, output_dir, attack_simulator)
    return {
        "dataset": dataset_name,
        "targeted_rows": len(targeted_rows),
        "written_samples": len(samples),
        "skipped_no_docs": skipped,
        "gold_answer_samples": sum(1 for sample in samples if sample.get("answers")),
        "qrels_doc_samples": qrels_used,
        "retrieval_only_samples": retrieval_used,
        "attack_counts": attack_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare aligned VeriRAG evaluation data")
    parser.add_argument("--source", required=True, help="BEIR data root")
    parser.add_argument("--poisonedrag-results", default="third_party/PoisonedRAG/results/adv_targeted_results")
    parser.add_argument("--beir-results", default="third_party/PoisonedRAG/results/beir_results")
    parser.add_argument("--output", default="./data_aligned_eval")
    parser.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "ms_marco"])
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    attack_simulator = AttackSimulator({"seed": args.seed})
    manifest = {
        "source_root": str(Path(args.source).resolve()),
        "poisonedrag_results": str(Path(args.poisonedrag_results).resolve()),
        "beir_results": str(Path(args.beir_results).resolve()),
        "max_docs": args.max_docs,
        "datasets": {},
    }

    output_dir = Path(args.output)
    for dataset_name in args.datasets:
        manifest["datasets"][dataset_name] = convert_dataset(
            dataset_name=dataset_name,
            source_root=Path(args.source),
            targeted_root=Path(args.poisonedrag_results),
            retrieval_root=Path(args.beir_results),
            output_dir=output_dir,
            max_docs=args.max_docs,
            attack_simulator=attack_simulator,
        )

    stats_path = output_dir / "alignment_manifest.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[Aligned] Manifest -> {stats_path}")


if __name__ == "__main__":
    main()
