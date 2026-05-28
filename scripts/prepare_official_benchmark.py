"""
Build an official-aligned VeriRAG benchmark directory with fixed attacks.

The input directory should already contain BEIR-backed samples with official QA
answers, as produced by scripts/import_official_answers.py. This script creates a
self-contained benchmark split:

  <output>/<dataset>_<split>.jsonl
  <output>/attacks/<dataset>_<attack_type>.jsonl
  <output>/official_benchmark_manifest.json

By default only eval_gold=true samples are retained. That keeps ACC/F1 and ASR
on the same scorable query set for external comparisons.
"""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from verirag.attack_simulator import AttackSimulator


DEFAULT_DATASETS = ["nq", "hotpotqa", "ms_marco"]
DEFAULT_SPLIT_MAP = {
    "nq": "test",
    "hotpotqa": "test",
    "ms_marco": "validation",
}
DEFAULT_ATTACK_TYPES = [
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


def stable_seed(base_seed: int, *parts: Any) -> int:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_samples(rows: List[Dict[str, Any]], *, include_weak: bool, max_samples: int) -> List[Dict[str, Any]]:
    selected = rows if include_weak else [row for row in rows if is_gold_sample(row)]
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def original_doc_texts(sample: Dict[str, Any]) -> List[str]:
    docs = sample.get("documents", [])
    texts = []
    if isinstance(docs, list):
        for doc in docs:
            if isinstance(doc, dict):
                text = str(doc.get("text", "")).strip()
            else:
                text = str(doc).strip()
            if text:
                texts.append(text)
    if not texts:
        fallback = str(sample.get("document") or sample.get("text") or "").strip()
        if fallback:
            texts.append(fallback)
    return texts


def ensure_target_answer(sample: Dict[str, Any]) -> str:
    target = str(sample.get("target_answer") or "").strip()
    if target:
        return target
    answers = sample.get("answers")
    if isinstance(answers, list) and answers:
        first = str(answers[0]).strip()
        if first:
            return f"NOT {first}"
    answer = str(sample.get("answer") or sample.get("ground_truth") or "").strip()
    return f"NOT {answer}" if answer else "unsupported answer"


def make_attack_row(
    dataset: str,
    split: str,
    sample: Dict[str, Any],
    attack_type: str,
    base_seed: int,
) -> Dict[str, Any]:
    sample_id = str(sample.get("id"))
    query_id = query_id_for(sample)
    target_answer = ensure_target_answer(sample)
    seed = stable_seed(base_seed, dataset, split, query_id, attack_type)
    simulator = AttackSimulator({"seed": seed})
    poisoned_documents = simulator.generate(
        query=str(sample.get("query") or sample.get("question") or ""),
        target_answer=target_answer,
        attack_type=attack_type,
        original_docs=original_doc_texts(sample),
    )
    return {
        "id": f"{sample_id}_{attack_type}",
        "sample_id": sample_id,
        "query_id": query_id,
        "dataset": dataset,
        "split": split,
        "attack_type": attack_type,
        "query": str(sample.get("query") or sample.get("question") or ""),
        "answers": sample.get("answers", []),
        "target_answer": target_answer,
        "poisoned_documents": poisoned_documents,
        "metadata": {
            "fixed_attack": True,
            "attack_source": "verirag_attack_simulator",
            "seed": seed,
            "eval_gold": is_gold_sample(sample),
            "source_sample_id": sample_id,
            "source_query_id": query_id,
            "num_poisoned_docs": len(poisoned_documents),
        },
    }


def write_dataset_benchmark(
    *,
    dataset: str,
    split: str,
    input_dir: Path,
    output_dir: Path,
    attack_types: List[str],
    include_weak: bool,
    max_samples: int,
    seed: int,
) -> Dict[str, Any]:
    input_path = input_dir / f"{dataset}_{split}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input split: {input_path}")

    rows = list(read_jsonl(input_path))
    selected = select_samples(rows, include_weak=include_weak, max_samples=max_samples)
    output_path = output_dir / f"{dataset}_{split}.jsonl"
    write_jsonl(output_path, selected)
    print(f"[Benchmark] {dataset}/{split}: {len(selected)} samples -> {output_path}")

    attack_counts: Dict[str, int] = {}
    for attack_type in attack_types:
        attack_path = output_dir / "attacks" / f"{dataset}_{attack_type}.jsonl"
        attack_rows = (
            make_attack_row(dataset, split, sample, attack_type, seed)
            for sample in selected
        )
        attack_counts[attack_type] = write_jsonl(attack_path, attack_rows)
        print(f"[Benchmark] {dataset}/{attack_type}: {attack_counts[attack_type]} fixed attacks -> {attack_path}")

    return {
        "dataset": dataset,
        "split": split,
        "source_path": str(input_path),
        "output_path": str(output_path),
        "source_rows": len(rows),
        "source_gold_rows": sum(1 for row in rows if is_gold_sample(row)),
        "source_weak_rows": sum(1 for row in rows if not is_gold_sample(row)),
        "written_samples": len(selected),
        "written_gold_samples": sum(1 for row in selected if is_gold_sample(row)),
        "written_weak_samples": sum(1 for row in selected if not is_gold_sample(row)),
        "attack_counts": attack_counts,
        "sample_ids_preview": [row.get("id") for row in selected[:5]],
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
    gold_only: bool,
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
    config["benchmark"] = {
        "fixed_attacks": True,
        "gold_only": gold_only,
        "data_dir": str(output_dir),
        "manifest": str(output_dir / "official_benchmark_manifest.json"),
    }
    config_out.parent.mkdir(parents=True, exist_ok=True)
    with config_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"[Benchmark] Eval config -> {config_out}")


def collect_file_hashes(output_dir: Path) -> Dict[str, str]:
    hashes = {}
    for path in sorted(output_dir.rglob("*.jsonl")):
        hashes[str(path.relative_to(output_dir))] = file_sha256(path)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official-aligned fixed-attack benchmark data")
    parser.add_argument("--input", default="./data_official_aligned", help="official-aligned data directory")
    parser.add_argument("--output", default="./data_official_benchmark", help="benchmark output directory")
    parser.add_argument("--base-config", default="experiments/official_aligned_eval_config.yaml")
    parser.add_argument("--config-out", default="experiments/official_benchmark_eval_config.yaml")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--attack-types", nargs="+", default=DEFAULT_ATTACK_TYPES)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all selected samples")
    parser.add_argument("--include-weak", action="store_true", help="include samples without official answers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    split_map = {dataset: DEFAULT_SPLIT_MAP.get(dataset, "test") for dataset in args.datasets}
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_data_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "datasets": {},
        "split_map": split_map,
        "attack_types": args.attack_types,
        "seed": args.seed,
        "gold_only": not args.include_weak,
        "max_samples_per_dataset": args.max_samples if args.max_samples > 0 else None,
        "format": {
            "samples": "<dataset>_<split>.jsonl",
            "attacks": "attacks/<dataset>_<attack_type>.jsonl",
            "fixed_attack_keys": ["sample_id", "query_id"],
        },
    }

    written_counts = []
    for dataset in args.datasets:
        stats = write_dataset_benchmark(
            dataset=dataset,
            split=split_map[dataset],
            input_dir=input_dir,
            output_dir=output_dir,
            attack_types=args.attack_types,
            include_weak=args.include_weak,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        manifest["datasets"][dataset] = stats
        written_counts.append(stats["written_samples"])

    manifest["file_sha256"] = collect_file_hashes(output_dir)
    manifest_path = output_dir / "official_benchmark_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[Benchmark] Manifest -> {manifest_path}")

    n_questions = max(written_counts) if written_counts else 0
    write_eval_config(
        base_config_path=Path(args.base_config) if args.base_config else None,
        config_out=Path(args.config_out),
        output_dir=output_dir,
        datasets=args.datasets,
        split_map=split_map,
        n_questions=n_questions,
        gold_only=not args.include_weak,
    )


if __name__ == "__main__":
    main()
