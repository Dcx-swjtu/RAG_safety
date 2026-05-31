#!/usr/bin/env python3
"""Summarize strict-ASR three-dataset eval outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


DATASET_ORDER = {"nq": 0, "hotpotqa": 1, "ms_marco": 2}
METHOD_ORDER = {
    "vanilla": 0,
    "instructrag": 1,
    "astuterag": 2,
    "trustrag": 3,
    "seconrag_lite": 4,
    "learned_scorer": 5,
    "ours": 6,
}
ATTACKS = [
    "poisonedrag_lm_targeted",
    "poisonedrag_hotflip",
    "garag",
    "tan_et_al",
    "advdecoding",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def baseline_rows(run_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(run_root.glob("baseline_*/*_baselines.json")):
        payload = load_json(path)
        meta = payload.get("meta", {})
        for result in payload.get("results", []):
            row = {
                "dataset": meta.get("dataset", ""),
                "method": result.get("method", ""),
                "acc": float(result.get("acc", 0.0)),
                "asr": float(result.get("avg_asr", 0.0)),
                "f1": float(result.get("f1", 0.0)),
                "source": str(path),
            }
            attacks = result.get("attack_results", {}) or {}
            for attack in ATTACKS:
                row[f"asr_{attack}"] = float((attacks.get(attack) or {}).get("attack_success_rate", 0.0))
            rows.append(row)
    return rows


def ours_rows(run_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(run_root.glob("ours_*/*_ours.json")):
        payload = load_json(path)
        row = {
            "dataset": payload.get("dataset", ""),
            "method": "ours",
            "acc": float(payload.get("acc", 0.0)),
            "asr": float(payload.get("avg_asr", 0.0)),
            "f1": float(payload.get("f1", 0.0)),
            "source": str(path),
        }
        attacks = payload.get("attack_breakdown", {}) or {}
        for attack in ATTACKS:
            row[f"asr_{attack}"] = float((attacks.get(attack) or {}).get("asr", 0.0))
        rows.append(row)
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = ["dataset", "method", "acc", "asr", "f1"] + [f"asr_{attack}" for attack in ATTACKS] + ["source"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "# Strict-ASR Three-Dataset Eval Summary",
        "",
        "| Dataset | Method | ACC % | ASR % | F1 % | LM-targeted ASR % | HotFlip ASR % | GARAG ASR % | TAN ASR % | AdvDecoding ASR % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {acc} | {asr} | {f1} | {lm} | {hotflip} | {garag} | {tan} | {adv} |".format(
                dataset=row["dataset"],
                method=row["method"],
                acc=pct(row["acc"]),
                asr=pct(row["asr"]),
                f1=pct(row["f1"]),
                lm=pct(row["asr_poisonedrag_lm_targeted"]),
                hotflip=pct(row["asr_poisonedrag_hotflip"]),
                garag=pct(row["asr_garag"]),
                tan=pct(row["asr_tan_et_al"]),
                adv=pct(row["asr_advdecoding"]),
            )
        )
    lines.extend(["", "ASR uses strict normalized target exact/phrase containment; ACC keeps the QA matcher.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_strict_asr_eval.py RUN_ROOT")
    run_root = Path(sys.argv[1]).resolve()
    rows = baseline_rows(run_root) + ours_rows(run_root)
    rows.sort(key=lambda row: (DATASET_ORDER.get(row["dataset"], 99), METHOD_ORDER.get(row["method"], 99)))
    if not rows:
        raise SystemExit(f"No result JSON files found under {run_root}")
    write_csv(rows, run_root / "strict_asr_summary.csv")
    write_md(rows, run_root / "strict_asr_summary.md")
    print(f"wrote {len(rows)} rows to {run_root / 'strict_asr_summary.md'}")


if __name__ == "__main__":
    main()
