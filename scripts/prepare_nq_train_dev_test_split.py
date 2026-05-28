#!/usr/bin/env python3
"""Prepare a leakage-safe NQ train/validation/test split.

The default protocol keeps the existing NQ-500 benchmark as held-out test and
splits the remaining official-aligned NQ gold samples into train/validation.
Attack rows are subset by query/sample id and written both as split-specific
files and as combined files for compatibility with older evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


DEFAULT_ATTACK_TYPES = [
    "poisonedrag",
    "oneshot",
    "refinerag",
    "semantic_chameleon",
    "adaptive",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def query_id(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) or {}
    return str(metadata.get("query_id") or row.get("query_id") or row.get("id") or "")


def sample_id(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or "")


def split_key(row: Dict[str, Any]) -> str:
    return query_id(row) or sample_id(row)


def stable_order(rows: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]) -> str:
        raw = f"{seed}|{split_key(row)}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    return sorted(rows, key=key)


def id_set(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    ids: Set[str] = set()
    for row in rows:
        for value in (row.get("id"), query_id(row)):
            if value:
                ids.add(str(value))
    return ids


def attack_matches(row: Dict[str, Any], keys: Set[str]) -> bool:
    return any(str(value) in keys for value in (row.get("sample_id"), row.get("query_id"), row.get("id")) if value)


def copy_attacks(
    *,
    source_dir: Path,
    output_dir: Path,
    attack_types: Sequence[str],
    split_rows: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, int]]:
    split_keys = {name: id_set(rows) for name, rows in split_rows.items()}
    stats: Dict[str, Dict[str, int]] = {name: {} for name in split_rows}
    combined: Dict[str, List[Dict[str, Any]]] = {attack_type: [] for attack_type in attack_types}

    for attack_type in attack_types:
        source_path = source_dir / "attacks" / f"nq_{attack_type}.jsonl"
        if not source_path.exists():
            continue
        rows = read_jsonl(source_path)
        for split_name, keys in split_keys.items():
            selected = [row for row in rows if attack_matches(row, keys)]
            stats[split_name][attack_type] = write_jsonl(
                output_dir / "attacks" / split_name / f"nq_{attack_type}.jsonl",
                selected,
            )
            combined[attack_type].extend(selected)

    for attack_type, rows in combined.items():
        write_jsonl(output_dir / "attacks" / f"nq_{attack_type}.jsonl", rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe NQ train/validation/test splits")
    parser.add_argument("--source", default="data_official_benchmark_full", help="official NQ full benchmark dir")
    parser.add_argument("--test-source", default="data_official_benchmark_500", help="held-out NQ test benchmark dir")
    parser.add_argument("--output", default="data_official_nq_split", help="output split dir")
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--dev-size", type=int, default=0, help="0 means use all non-train remaining rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack-types", nargs="+", default=DEFAULT_ATTACK_TYPES)
    args = parser.parse_args()

    source_dir = Path(args.source)
    test_source_dir = Path(args.test_source)
    output_dir = Path(args.output)
    full_rows = read_jsonl(source_dir / "nq_test.jsonl")
    test_rows = read_jsonl(test_source_dir / "nq_test.jsonl")
    test_ids = id_set(test_rows)

    remaining = [row for row in full_rows if not any(str(value) in test_ids for value in (row.get("id"), query_id(row)) if value)]
    ordered = stable_order(remaining, args.seed)
    train_rows = ordered[: args.train_size]
    rest = ordered[args.train_size :]
    validation_rows = rest if args.dev_size <= 0 else rest[: args.dev_size]

    split_rows = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_counts = {
        "train": write_jsonl(output_dir / "nq_train.jsonl", train_rows),
        "validation": write_jsonl(output_dir / "nq_validation.jsonl", validation_rows),
        "test": write_jsonl(output_dir / "nq_test.jsonl", test_rows),
    }
    attack_counts = copy_attacks(
        source_dir=source_dir,
        output_dir=output_dir,
        attack_types=args.attack_types,
        split_rows=split_rows,
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source_dir.resolve()),
        "test_source": str(test_source_dir.resolve()),
        "output": str(output_dir.resolve()),
        "seed": args.seed,
        "protocol": "NQ full official-gold rows minus held-out NQ-500 test; deterministic train/validation split by query id hash.",
        "sample_counts": sample_counts,
        "attack_counts": attack_counts,
        "attack_types": list(args.attack_types),
        "held_out_test_preserves_existing_nq500": True,
        "files": {
            "train": "nq_train.jsonl",
            "validation": "nq_validation.jsonl",
            "test": "nq_test.jsonl",
            "split_attacks": "attacks/<split>/nq_<attack_type>.jsonl",
            "combined_attacks": "attacks/nq_<attack_type>.jsonl",
        },
    }
    with (output_dir / "nq_split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
