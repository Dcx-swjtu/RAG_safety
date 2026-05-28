#!/usr/bin/env python3
"""Build train/validation/test NQ splits with official mixed attacks.

This keeps the project split protocol:

  data_official_mixed_attack_nq_split/
    nq_train.jsonl
    nq_validation.jsonl
    nq_test.jsonl
    attacks/train/nq_<attack_type>.jsonl
    attacks/validation/nq_<attack_type>.jsonl
    attacks/test/nq_<attack_type>.jsonl

The attack conversion logic is shared with
`build_official_mixed_attack_nq500.py`, so training and test data use the same
official-artifact backend and the same VeriRAG fixed-attack schema.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_official_mixed_attack_nq500 import (  # noqa: E402
    ATTACK_TYPES,
    build_clean_rows,
    build_gmtp_attack,
    build_poisonedrag_lm,
    build_ragdefender_attack,
    read_jsonl,
    write_jsonl,
)


def merge_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)


def write_dataset_card(output: Path, manifest: Dict[str, Any]) -> None:
    attack_lines = "\n".join(f"- `{attack}`" for attack in ATTACK_TYPES)
    split_lines = "\n".join(
        f"- {split}: {info['num_questions']} questions"
        for split, info in manifest["splits"].items()
    )
    coverage_lines: List[str] = []
    for attack in ATTACK_TYPES:
        coverage_lines.append(f"- `{attack}`:")
        for key, value in manifest["aggregate_attack_source_stats"][attack].items():
            coverage_lines.append(f"  - {key}: {value}")
    content = f"""# Official Mixed-Attack NQ Train/Dev/Test

## Purpose

This dataset is the training-compatible version of
`data_official_mixed_attack_nq500`. It preserves the leakage-safe NQ split while
using official/public attack artifacts for fixed attack construction.

## Splits

{split_lines}

## Attack Types

{attack_lines}

## Clean Contexts

Clean contexts are rebuilt from the PoisonedRAG official Contriever top-5
retrieval file when available. The original qrels documents are used only as a
fallback.

## Coverage

Public official artifacts do not cover every query in the project split. Missing
rows are filled deterministically from the same official artifact pool, and the
source mode is recorded per row and in the manifest.

{chr(10).join(coverage_lines)}

## Main Use

Use this directory for scorer/PPO training:

```text
data_dir: data_official_mixed_attack_nq_split
split: train
```

Use `data_official_mixed_attack_nq500` for the root-level held-out evaluation
config unless the evaluator is explicitly made split-aware.

"""
    (output / "DATASET_CARD.md").write_text(content, encoding="utf-8")


def build_one_split(
    split: str,
    source_path: Path,
    output: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    samples = read_jsonl(source_path)
    if args.max_rows and len(samples) > args.max_rows:
        samples = samples[: args.max_rows]

    clean_rows, clean_stats = build_clean_rows(
        samples,
        corpus_path=Path(args.corpus),
        retrieval_path=Path(args.poisonedrag_root) / "results" / "beir_results" / "nq-contriever.json",
        top_k=args.top_k,
    )
    write_jsonl(output / f"nq_{split}.jsonl", clean_rows)

    split_attack_dir = output / "attacks" / split
    split_attack_dir.mkdir(parents=True, exist_ok=True)

    attack_stats: Dict[str, Dict[str, int]] = {}

    rows, stats = build_poisonedrag_lm(samples, Path(args.poisonedrag_root), args.adv_per_query)
    write_jsonl(split_attack_dir / "nq_poisonedrag_lm_targeted.jsonl", rows)
    attack_stats["poisonedrag_lm_targeted"] = stats

    rows, stats = build_gmtp_attack(
        samples,
        Path(args.gmtp_root)
        / "data"
        / "poisoned_documents"
        / "poisonedrag"
        / "hotflip"
        / "contriever"
        / "nq-200.json",
        "poisonedrag_hotflip",
        "GMTP/PoisonedRAG-HotFlip",
        ["poisoned_docs", "poisoned_texts"],
        args.adv_per_query,
    )
    write_jsonl(split_attack_dir / "nq_poisonedrag_hotflip.jsonl", rows)
    attack_stats["poisonedrag_hotflip"] = stats

    rows, stats = build_ragdefender_attack(
        samples,
        Path(args.ragdefender_root) / "artifacts" / "GARAG" / "garag_nq.json",
        "garag",
        "RAGDefender/GARAG",
    )
    write_jsonl(split_attack_dir / "nq_garag.jsonl", rows)
    attack_stats["garag"] = stats

    rows, stats = build_ragdefender_attack(
        samples,
        Path(args.ragdefender_root) / "artifacts" / "tan" / "tan_nq.json",
        "tan_et_al",
        "RAGDefender/Tan-et-al",
    )
    write_jsonl(split_attack_dir / "nq_tan_et_al.jsonl", rows)
    attack_stats["tan_et_al"] = stats

    rows, stats = build_gmtp_attack(
        samples,
        Path(args.gmtp_root)
        / "data"
        / "poisoned_documents"
        / "advdecoding"
        / "trigger_append"
        / "contriever"
        / "nq-200.json",
        "advdecoding",
        "GMTP/Adversarial-Decoding",
        ["poisoned_docs", "poisoned_texts", "gen_atk"],
        args.adv_per_query,
    )
    write_jsonl(split_attack_dir / "nq_advdecoding.jsonl", rows)
    attack_stats["advdecoding"] = stats

    info = {
        "source": str(source_path),
        "num_questions": len(samples),
        "clean_context_stats": clean_stats,
        "attack_dir": str(split_attack_dir),
        "attack_counts": {attack: len(samples) for attack in ATTACK_TYPES},
        "attack_source_stats": attack_stats,
    }
    return info, attack_stats


def copy_test_root_attacks(output: Path) -> None:
    test_dir = output / "attacks" / "test"
    root_dir = output / "attacks"
    for attack in ATTACK_TYPES:
        src = test_dir / f"nq_{attack}.jsonl"
        dst = root_dir / f"nq_{attack}.jsonl"
        if src.exists():
            shutil.copyfile(src, dst)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to rebuild")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    split_sources = {
        "train": Path(args.train_source),
        "validation": Path(args.validation_source),
        "test": Path(args.test_source),
    }

    splits: Dict[str, Any] = {}
    aggregate_attack_stats: Dict[str, Dict[str, int]] = {attack: {} for attack in ATTACK_TYPES}
    for split, source_path in split_sources.items():
        info, attack_stats = build_one_split(split, source_path, output, args)
        splits[split] = info
        for attack, stats in attack_stats.items():
            merge_counts(aggregate_attack_stats[attack], stats)

    if args.copy_test_attacks_to_root:
        copy_test_root_attacks(output)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": "data_official_mixed_attack_nq_split",
        "description": (
            "Leakage-safe NQ train/validation/test split with official/public "
            "mixed attack artifacts converted to VeriRAG fixed-attack schema."
        ),
        "dataset": "nq",
        "attack_types": ATTACK_TYPES,
        "splits": splits,
        "aggregate_attack_source_stats": aggregate_attack_stats,
        "settings": {
            "top_k": args.top_k,
            "adv_per_query": args.adv_per_query,
            "copy_test_attacks_to_root": args.copy_test_attacks_to_root,
            "official_artifact_fill_policy": (
                "query-id/query-text match when available; deterministic same-artifact pool fill "
                "for official artifacts whose public release does not cover a split query"
            ),
        },
        "schema": {
            "split_files": "nq_<split>.jsonl",
            "split_attacks": "attacks/<split>/nq_<attack_type>.jsonl",
            "root_test_attack_aliases": "attacks/nq_<attack_type>.jsonl",
        },
    }
    with (output / "official_mixed_attack_split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    write_dataset_card(output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    rag_root = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", default=str(root / "data_official_nq_split" / "nq_train.jsonl"))
    parser.add_argument("--validation-source", default=str(root / "data_official_nq_split" / "nq_validation.jsonl"))
    parser.add_argument("--test-source", default=str(root / "data_official_nq_split" / "nq_test.jsonl"))
    parser.add_argument("--output", default=str(root / "data_official_mixed_attack_nq_split"))
    parser.add_argument("--corpus", default=str(rag_root / "data" / "nq" / "corpus.jsonl"))
    parser.add_argument("--poisonedrag-root", default=str(rag_root / "data_process" / "PoisonedRAG"))
    parser.add_argument("--ragdefender-root", default=str(rag_root / "data_process" / "RAGDefender"))
    parser.add_argument("--gmtp-root", default=str(rag_root / "data_process" / "GMTP"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--adv-per-query", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--copy-test-attacks-to-root", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
