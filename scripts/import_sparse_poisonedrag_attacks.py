"""
Import Sparse-Document-Attention-RAG / PoisonedRAG attack CSVs into VeriRAG format.

This aligns paper-provided malicious documents with our official-answer BEIR
samples by normalized query text. The resulting benchmark keeps BEIR documents
and official QA answers for clean ACC, and uses the CSV false answer plus
malicious documents for fixed-attack ASR.
"""

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


DEFAULT_DATASETS = ["nq", "hotpotqa"]
DEFAULT_SPLIT_MAP = {
    "nq": "test",
    "hotpotqa": "test",
}
DEFAULT_CSV_FILES = {
    "nq": "poisonedRAG_attack_results_GPT4_NQ_5_mal_docs_per_query.csv",
    "hotpotqa": "poisonedRAG_attack_results_GPT4_hotpotQA_5_mal_docs_per_query.csv",
    "triviaqa": "poisonedRAG_attack_results_GPT4_triviaQA_5_mal_docs_per_query.csv",
}


def normalize_query(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def is_gold_sample(row: Dict[str, Any]) -> bool:
    metadata = row.get("metadata", {}) or {}
    if metadata.get("eval_gold") is False:
        return False
    if metadata.get("answer_source") == "weak_document_label":
        return False
    answers = row.get("answers")
    if isinstance(answers, list) and any(str(answer).strip() for answer in answers):
        return True
    return bool(str(row.get("answer") or row.get("ground_truth") or "").strip())


def query_id_for(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) or {}
    return str(metadata.get("query_id") or row.get("id") or "")


def parse_answers(raw: Any) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    if str(parsed).strip():
        return [str(parsed)]
    return []


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_hashes(output_dir: Path) -> Dict[str, str]:
    hashes = {}
    for path in sorted(output_dir.rglob("*.jsonl")):
        hashes[str(path.relative_to(output_dir))] = file_sha256(path)
    return hashes


def load_official_samples(path: Path, *, gold_only: bool) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    samples: Dict[str, Dict[str, Any]] = {}
    duplicate_keys = 0
    total = 0
    gold = 0
    kept = 0
    for row in read_jsonl(path):
        total += 1
        if is_gold_sample(row):
            gold += 1
        elif gold_only:
            continue
        key = normalize_query(row.get("query") or row.get("question"))
        if not key:
            continue
        if key in samples:
            duplicate_keys += 1
            continue
        samples[key] = row
        kept += 1
    return samples, {
        "official_rows": total,
        "official_gold_rows": gold,
        "official_rows_kept_for_matching": kept,
        "duplicate_normalized_queries": duplicate_keys,
    }


def load_attack_groups(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    groups: Dict[str, Dict[str, Any]] = {}
    rows = 0
    inconsistent_false_answers = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            query = row.get("query") or ""
            key = normalize_query(query)
            if not key:
                continue
            group = groups.setdefault(
                key,
                {
                    "query": query,
                    "paper_query_id": row.get("query_id") or "",
                    "paper_ground_truth_answers": parse_answers(row.get("ground_truth_answers")),
                    "false_answer": str(row.get("false_answer") or "").strip(),
                    "malicious_documents": [],
                },
            )
            false_answer = str(row.get("false_answer") or "").strip()
            if false_answer and group["false_answer"] and false_answer != group["false_answer"]:
                inconsistent_false_answers += 1
            elif false_answer and not group["false_answer"]:
                group["false_answer"] = false_answer

            doc = str(row.get("malicious_document") or "").strip()
            if doc and doc not in group["malicious_documents"]:
                group["malicious_documents"].append(doc)

    return groups, {
        "csv_rows": rows,
        "csv_queries": len(groups),
        "inconsistent_false_answer_rows": inconsistent_false_answers,
    }


def make_sample_row(
    official: Dict[str, Any],
    attack_group: Dict[str, Any],
    *,
    dataset: str,
    attack_type: str,
    source_csv: Path,
) -> Dict[str, Any]:
    row = copy.deepcopy(official)
    row["target_answer"] = attack_group.get("false_answer") or row.get("target_answer") or "unsupported answer"
    metadata = row.setdefault("metadata", {})
    metadata.update(
        {
            "paper_attack_aligned": True,
            "paper_benchmark": "Sparse-Document-Attention-RAG",
            "paper_attack_family": "PoisonedRAG",
            "paper_attack_generator": "GPT-4",
            "paper_attack_type": attack_type,
            "paper_attack_source_file": str(source_csv),
            "paper_query": attack_group.get("query", ""),
            "paper_query_id": attack_group.get("paper_query_id", ""),
            "paper_ground_truth_answers": attack_group.get("paper_ground_truth_answers", []),
            "paper_false_answer": attack_group.get("false_answer", ""),
            "paper_num_malicious_docs": len(attack_group.get("malicious_documents", [])),
            "dataset": dataset,
        }
    )
    return row


def make_attack_row(
    sample: Dict[str, Any],
    attack_group: Dict[str, Any],
    *,
    dataset: str,
    split: str,
    attack_type: str,
    source_csv: Path,
) -> Dict[str, Any]:
    sample_id = str(sample.get("id"))
    query_id = query_id_for(sample)
    target_answer = attack_group.get("false_answer") or sample.get("target_answer") or "unsupported answer"
    return {
        "id": f"{sample_id}_{attack_type}",
        "sample_id": sample_id,
        "query_id": query_id,
        "dataset": dataset,
        "split": split,
        "attack_type": attack_type,
        "query": sample.get("query") or sample.get("question") or "",
        "answers": sample.get("answers", []),
        "target_answer": target_answer,
        "poisoned_documents": list(attack_group.get("malicious_documents", [])),
        "metadata": {
            "fixed_attack": True,
            "attack_source": "sparse_document_attention_rag_poisonedrag_gpt4_csv",
            "source_csv": str(source_csv),
            "paper_query_id": attack_group.get("paper_query_id", ""),
            "paper_ground_truth_answers": attack_group.get("paper_ground_truth_answers", []),
            "paper_false_answer": attack_group.get("false_answer", ""),
            "num_poisoned_docs": len(attack_group.get("malicious_documents", [])),
            "eval_gold": is_gold_sample(sample),
            "source_sample_id": sample_id,
            "source_query_id": query_id,
        },
    }


def write_dataset(
    *,
    dataset: str,
    split: str,
    official_dir: Path,
    sparse_data_dir: Path,
    output_dir: Path,
    attack_type: str,
    gold_only: bool,
    max_samples: int,
) -> Dict[str, Any]:
    official_path = official_dir / f"{dataset}_{split}.jsonl"
    if not official_path.exists():
        raise FileNotFoundError(f"Missing official-aligned split: {official_path}")
    csv_path = sparse_data_dir / DEFAULT_CSV_FILES[dataset]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Sparse/PoisonedRAG attack CSV: {csv_path}")

    official_by_query, official_stats = load_official_samples(official_path, gold_only=gold_only)
    attack_groups, csv_stats = load_attack_groups(csv_path)

    sample_rows: List[Dict[str, Any]] = []
    attack_rows: List[Dict[str, Any]] = []
    unmatched_queries: List[str] = []
    for key, group in attack_groups.items():
        official = official_by_query.get(key)
        if official is None:
            if len(unmatched_queries) < 20:
                unmatched_queries.append(group.get("query", ""))
            continue
        sample = make_sample_row(
            official,
            group,
            dataset=dataset,
            attack_type=attack_type,
            source_csv=csv_path,
        )
        sample_rows.append(sample)
        attack_rows.append(
            make_attack_row(
                sample,
                group,
                dataset=dataset,
                split=split,
                attack_type=attack_type,
                source_csv=csv_path,
            )
        )
        if max_samples > 0 and len(sample_rows) >= max_samples:
            break

    sample_path = output_dir / f"{dataset}_{split}.jsonl"
    attack_path = output_dir / "attacks" / f"{dataset}_{attack_type}.jsonl"
    write_jsonl(sample_path, sample_rows)
    write_jsonl(attack_path, attack_rows)

    return {
        "dataset": dataset,
        "split": split,
        "official_path": str(official_path),
        "attack_csv": str(csv_path),
        **official_stats,
        **csv_stats,
        "written_samples": len(sample_rows),
        "written_attacks": len(attack_rows),
        "gold_only": gold_only,
        "max_samples": max_samples if max_samples > 0 else None,
        "unmatched_csv_queries": max(csv_stats["csv_queries"] - len(sample_rows), 0),
        "unmatched_queries_preview": unmatched_queries,
        "sample_ids_preview": [row.get("id") for row in sample_rows[:5]],
    }


def load_base_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_eval_config(
    *,
    base_config_path: Optional[Path],
    config_out: Path,
    output_dir: Path,
    datasets: List[str],
    split_map: Dict[str, str],
    n_questions: int,
    attack_type: str,
    gold_only: bool,
    manifest_path: Path,
) -> None:
    config = copy.deepcopy(load_base_config(base_config_path))
    config.setdefault("data", {})
    config.setdefault("evaluation", {})
    config["data"]["data_dir"] = str(output_dir)
    config["data"]["datasets"] = list(datasets)
    config["evaluation"]["datasets"] = list(datasets)
    config["evaluation"]["split_map"] = split_map
    config["evaluation"]["n_questions"] = n_questions
    config["evaluation"]["require_fixed_attacks"] = True
    config["evaluation"]["attack_types"] = [attack_type]
    config["benchmark"] = {
        "fixed_attacks": True,
        "gold_only": gold_only,
        "data_dir": str(output_dir),
        "manifest": str(manifest_path),
        "paper_attack_source": "Sparse-Document-Attention-RAG PoisonedRAG GPT-4 CSV",
    }
    config_out.parent.mkdir(parents=True, exist_ok=True)
    with config_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Sparse/PoisonedRAG attack CSVs into VeriRAG fixed-attack format")
    parser.add_argument("--official-data", default="./data_official_aligned")
    parser.add_argument("--sparse-data", default="./third_party/Sparse-Document-Attention-RAG/data")
    parser.add_argument("--output", default="./data_sparse_poisonedrag_aligned")
    parser.add_argument("--base-config", default="experiments/official_aligned_eval_config.yaml")
    parser.add_argument("--config-out", default="experiments/sparse_poisonedrag_eval_config.yaml")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--attack-type", default="poisonedrag")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use every matched sample")
    parser.add_argument("--include-weak", action="store_true", help="include samples without official answers")
    args = parser.parse_args()

    official_dir = Path(args.official_data)
    sparse_data_dir = Path(args.sparse_data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_map = {dataset: DEFAULT_SPLIT_MAP.get(dataset, "test") for dataset in args.datasets}
    gold_only = not args.include_weak

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_official_data_dir": str(official_dir.resolve()),
        "source_sparse_data_dir": str(sparse_data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "datasets": {},
        "split_map": split_map,
        "attack_types": [args.attack_type],
        "gold_only": gold_only,
        "format": {
            "samples": "<dataset>_<split>.jsonl",
            "attacks": "attacks/<dataset>_<attack_type>.jsonl",
            "fixed_attack_keys": ["sample_id", "query_id"],
        },
        "alignment_rule": "normalized exact query text",
    }

    written_counts = []
    for dataset in args.datasets:
        if dataset not in DEFAULT_CSV_FILES:
            raise ValueError(f"Unsupported dataset for bundled Sparse/PoisonedRAG CSV import: {dataset}")
        stats = write_dataset(
            dataset=dataset,
            split=split_map[dataset],
            official_dir=official_dir,
            sparse_data_dir=sparse_data_dir,
            output_dir=output_dir,
            attack_type=args.attack_type,
            gold_only=gold_only,
            max_samples=args.max_samples,
        )
        manifest["datasets"][dataset] = stats
        written_counts.append(stats["written_samples"])
        print(
            f"[SparseImport] {dataset}/{split_map[dataset]}: "
            f"{stats['written_samples']} matched samples, "
            f"{stats['written_attacks']} fixed attacks"
        )

    manifest["file_sha256"] = collect_file_hashes(output_dir)
    manifest_path = output_dir / "sparse_poisonedrag_alignment_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[SparseImport] Manifest -> {manifest_path}")

    n_questions = max(written_counts) if written_counts else 0
    write_eval_config(
        base_config_path=Path(args.base_config) if args.base_config else None,
        config_out=Path(args.config_out),
        output_dir=output_dir,
        datasets=args.datasets,
        split_map=split_map,
        n_questions=n_questions,
        attack_type=args.attack_type,
        gold_only=gold_only,
        manifest_path=manifest_path,
    )
    print(f"[SparseImport] Eval config -> {args.config_out}")


if __name__ == "__main__":
    main()
