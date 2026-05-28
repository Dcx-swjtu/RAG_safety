#!/usr/bin/env python3
"""Fine-grained ACC/ASR diagnostics on the official-mixed NQ benchmark.

This script is intentionally diagnostic-only. It does not change the model
story or training code; it records per-sample evidence decisions so clean
accuracy loss can be separated from attack leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from evaluate_rag_defense_baselines import (  # noqa: E402
    BaselineMethods,
    BaselineOutput,
    check_answer,
    extract_poisoned_texts,
    generate_answer,
    get_answers,
    load_attacks,
    load_rows,
)
from verirag.attack_simulator import AttackSimulator  # noqa: E402
from verirag.claim_extractor import ClaimExtractor  # noqa: E402
from verirag.cross_validator import CrossValidator  # noqa: E402
from verirag.defense_orchestrator import DefenseOrchestrator, FinalAnswerStatus  # noqa: E402
from verirag.generator import QwenGenerator  # noqa: E402
from verirag.policy_network import VerificationPolicyNetwork  # noqa: E402
from verirag.text_features import get_doc_text  # noqa: E402


DEFAULT_ATTACK_TYPES = [
    "poisonedrag_lm_targeted",
    "poisonedrag_hotflip",
    "garag",
    "tan_et_al",
    "advdecoding",
]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def doc_id(doc: Dict[str, Any], idx: int) -> str:
    return str(doc.get("doc_id") or doc.get("id") or f"doc_{idx}")


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def f1_from_acc_asr(acc: float, asr: float) -> float:
    dsr = 1.0 - asr
    return 2.0 * acc * dsr / max(acc + dsr, 1e-8)


def build_generator(args: argparse.Namespace) -> QwenGenerator:
    return QwenGenerator(
        model_path=args.model_path,
        backend=args.backend,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_model=args.backend != "fallback",
    )


def build_baseline_methods(config: Dict[str, Any]) -> BaselineMethods:
    defense_cfg = config.get("defense", {})
    baseline_cfg = {
        "feature_extractor": defense_cfg.get("feature_extractor", {}),
        "doc_scorer": defense_cfg.get("doc_scorer", {}),
        "learned_threshold": defense_cfg.get("doc_filter_threshold", 0.50),
        "min_docs": defense_cfg.get("min_docs_after_filter", 1),
        "max_drop_fraction": defense_cfg.get("max_doc_drop_fraction", 0.85),
    }
    return BaselineMethods(baseline_cfg)


def build_orchestrator(config: Dict[str, Any], generator: QwenGenerator, args: argparse.Namespace) -> DefenseOrchestrator:
    defense_config = dict(config.get("defense", {}))
    if args.disable_caf:
        defense_config["enable_conflict_aware_generation"] = False
    if args.disable_nq_policy:
        defense_config["enable_nq_doc_policy"] = False
    if args.disable_doc_scorer:
        defense_config["enable_doc_scorer"] = False
        defense_config["enable_nq_doc_policy"] = False

    policy_net = VerificationPolicyNetwork(config=config.get("model", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = policy_net.to(device)
    policy_net.eval()
    return DefenseOrchestrator(
        policy_network=policy_net,
        claim_extractor=ClaimExtractor(config=config.get("claim_extractor", {})),
        cross_validator=CrossValidator(config=config.get("cross_validator", {})),
        base_llm=generator,
        config=defense_config,
    )


def attack_docs_for(row: Dict[str, Any], attack_row: Dict[str, Any], attack_type: str) -> List[Dict[str, Any]]:
    texts = extract_poisoned_texts(attack_row.get("poisoned_documents", []))
    return [
        {"doc_id": f"attack_{attack_type}_{idx}", "text": text, "source": f"attack:{attack_type}"}
        for idx, text in enumerate(texts)
    ]


def row_key(row: Dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata", {}) or {}
    return str(row.get("id")), str(metadata.get("query_id"))


def find_attack_row(attacks: Dict[str, Dict[str, Any]], row: Dict[str, Any]) -> Dict[str, Any]:
    sample_id, query_id = row_key(row)
    attack_row = attacks.get(sample_id) or attacks.get(query_id)
    if attack_row is None:
        raise KeyError(f"Missing attack row: sample_id={sample_id}, query_id={query_id}")
    return attack_row


def serialize_scores(doc_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = []
    for row in doc_scores or []:
        keep.append(
            {
                "doc_id": row.get("doc_id"),
                "attack_prob": row.get("attack_prob"),
                "support_score": row.get("support_score"),
                "conflict_score": row.get("conflict_score"),
                "keep_prob": row.get("nq_doc_policy_keep_prob"),
                "kept": row.get("nq_doc_policy_kept"),
                "reasons": row.get("reasons", []),
            }
        )
    return keep


def eval_oracle_case(
    generator: QwenGenerator,
    row: Dict[str, Any],
    attack_row: Dict[str, Any] | None,
    attack_type: str | None,
) -> Dict[str, Any]:
    query = row.get("query") or row.get("question") or ""
    clean_docs = list(row.get("documents", []))
    attack_docs = attack_docs_for(row, attack_row, attack_type) if attack_row and attack_type else []
    docs_for_generation = clean_docs
    answer = generator.generate_answer(query, [get_doc_text(doc) for doc in docs_for_generation[:8]])
    return {
        "answer": answer,
        "status": "answered",
        "dropped_doc_ids": [doc_id(doc, idx + len(clean_docs)) for idx, doc in enumerate(attack_docs)],
        "doc_scores": [],
        "detected": bool(attack_docs),
        "policy_action": "ORACLE_KEEP_CLEAN",
        "trace": ["Oracle keeps clean evidence and drops injected attack docs."],
    }


def eval_baseline_case(
    method: str,
    baseline_methods: BaselineMethods,
    generator: QwenGenerator,
    row: Dict[str, Any],
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    query = row.get("query") or row.get("question") or ""
    out: BaselineOutput = baseline_methods.apply(method, query, docs)
    answer = generate_answer(generator, out)
    return {
        "answer": answer,
        "status": out.status,
        "dropped_doc_ids": list(out.dropped_doc_ids or []),
        "doc_scores": [],
        "detected": out.status == "rejected",
        "policy_action": method,
        "trace": [],
    }


def eval_ours_case(
    orchestrator: DefenseOrchestrator,
    row: Dict[str, Any],
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    query = row.get("query") or row.get("question") or ""
    result = orchestrator.defend(query, docs)
    return {
        "answer": result.final_answer,
        "status": result.status.value if isinstance(result.status, FinalAnswerStatus) else str(result.status),
        "dropped_doc_ids": list(result.filtered_doc_ids or []),
        "doc_scores": serialize_scores(result.doc_scores),
        "detected": result.status == FinalAnswerStatus.REJECTED or bool(result.detected_attacks),
        "policy_action": result.policy_action,
        "trace": list(result.execution_trace or []),
        "risk_indicators": list(result.risk_indicators or []),
        "module_timings": dict(result.module_timings or {}),
    }


def make_trace_row(
    method: str,
    row: Dict[str, Any],
    case_type: str,
    attack_type: str | None,
    docs: List[Dict[str, Any]],
    attack_doc_ids: set[str],
    eval_result: Dict[str, Any],
    target_answer: str | None,
) -> Dict[str, Any]:
    answers = get_answers(row)
    answer = eval_result["answer"]
    dropped = set(str(x) for x in eval_result.get("dropped_doc_ids", []))
    clean_doc_ids = {doc_id(doc, idx) for idx, doc in enumerate(row.get("documents", []))}
    dropped_clean = sorted(dropped & clean_doc_ids)
    dropped_attack = sorted(dropped & attack_doc_ids)
    correct = check_answer(answer, answers)
    attack_success = bool(target_answer and check_answer(answer, target_answer))
    rejected = str(eval_result.get("status", "")).lower() == "rejected"
    detected = bool(eval_result.get("detected", False) or rejected or dropped_attack)
    metadata = row.get("metadata", {}) or {}
    return {
        "method": method,
        "case_type": case_type,
        "attack_type": attack_type,
        "sample_id": row.get("id"),
        "query_id": metadata.get("query_id"),
        "query": row.get("query") or row.get("question") or "",
        "answers": answers,
        "target_answer": target_answer,
        "answer": answer,
        "answer_correct": correct,
        "attack_success": attack_success,
        "status": eval_result.get("status"),
        "rejected": rejected,
        "detected": detected,
        "num_docs": len(docs),
        "num_clean_docs": len(row.get("documents", [])),
        "num_attack_docs": len(attack_doc_ids),
        "dropped_doc_ids": sorted(dropped),
        "dropped_clean_doc_ids": dropped_clean,
        "dropped_attack_doc_ids": dropped_attack,
        "clean_drop_ratio": len(dropped_clean) / max(len(row.get("documents", [])), 1),
        "attack_drop_ratio": len(dropped_attack) / max(len(attack_doc_ids), 1) if attack_doc_ids else 0.0,
        "doc_scores": eval_result.get("doc_scores", []),
        "policy_action": eval_result.get("policy_action"),
        "trace": eval_result.get("trace", []),
        "risk_indicators": eval_result.get("risk_indicators", []),
        "module_timings": eval_result.get("module_timings", {}),
    }


def summarize_trace(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean = [row for row in rows if row["case_type"] == "clean"]
    attacks = [row for row in rows if row["case_type"] == "attack"]
    acc = mean([float(row["answer_correct"]) for row in clean])
    asr = mean([float(row["attack_success"]) for row in attacks])
    attack_breakdown: Dict[str, Dict[str, Any]] = {}
    for attack_type in sorted({row["attack_type"] for row in attacks}):
        sub = [row for row in attacks if row["attack_type"] == attack_type]
        attack_breakdown[str(attack_type)] = {
            "total": len(sub),
            "asr": mean([float(row["attack_success"]) for row in sub]),
            "detected": mean([float(row["detected"]) for row in sub]),
            "attack_drop_ratio": mean([float(row["attack_drop_ratio"]) for row in sub]),
            "attack_success_detected": int(sum(1 for row in sub if row["attack_success"] and row["detected"])),
        }
    return {
        "clean_total": len(clean),
        "attack_total": len(attacks),
        "acc": acc,
        "avg_asr": asr,
        "f1": f1_from_acc_asr(acc, asr),
        "fpr_rejected": mean([float(row["rejected"]) for row in clean]),
        "clean_wrong": int(sum(1 for row in clean if not row["answer_correct"])),
        "clean_wrong_non_rejected": int(sum(1 for row in clean if not row["answer_correct"] and not row["rejected"])),
        "clean_rejected": int(sum(1 for row in clean if row["rejected"])),
        "clean_drop_any": mean([float(bool(row["dropped_clean_doc_ids"])) for row in clean]),
        "clean_drop_ratio": mean([float(row["clean_drop_ratio"]) for row in clean]),
        "attack_detected": mean([float(row["detected"]) for row in attacks]),
        "attack_drop_ratio": mean([float(row["attack_drop_ratio"]) for row in attacks]),
        "attack_success_detected": int(sum(1 for row in attacks if row["attack_success"] and row["detected"])),
        "attack_breakdown": attack_breakdown,
    }


def write_outputs(trace_rows: List[Dict[str, Any]], summary: Dict[str, Any], output: str, trace_output: str) -> None:
    out_path = Path(output)
    trace_path = Path(trace_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md_path = out_path.with_suffix(".md")
    lines = [
        f"# Official-Mixed ACC/ASR Diagnostic: {summary['method']}",
        "",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- ACC: {summary['acc']:.4f}",
        f"- ASR: {summary['avg_asr']:.4f}",
        f"- F1: {summary['f1']:.4f}",
        f"- FPR rejected: {summary['fpr_rejected']:.4f}",
        f"- Clean wrong: {summary['clean_wrong']}",
        f"- Clean wrong non-rejected: {summary['clean_wrong_non_rejected']}",
        f"- Clean rejected: {summary['clean_rejected']}",
        f"- Clean drop any: {summary['clean_drop_any']:.4f}",
        f"- Clean drop ratio: {summary['clean_drop_ratio']:.4f}",
        f"- Attack detected: {summary['attack_detected']:.4f}",
        f"- Attack drop ratio: {summary['attack_drop_ratio']:.4f}",
        f"- Attack success but detected: {summary['attack_success_detected']}",
        "",
        "## Attack Breakdown",
        "",
        "| Attack | ASR | Detected | AttackDrop | SuccessDetected |",
        "|---|---:|---:|---:|---:|",
    ]
    for attack_type, stats in summary["attack_breakdown"].items():
        lines.append(
            f"| {attack_type} | {stats['asr']:.4f} | {stats['detected']:.4f} | "
            f"{stats['attack_drop_ratio']:.4f} | {stats['attack_success_detected']} |"
        )
    lines.extend(["", f"Trace: `{trace_path}`", f"JSON: `{out_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose ACC/ASR failure modes on official-mixed NQ.")
    parser.add_argument("--config", default="configs/main/official_mixed_attack_nq500_qwen_reward_official_mixed_trained.yaml")
    parser.add_argument("--method", required=True, choices=[
        "oracle",
        "vanilla",
        "learned_scorer",
        "trustrag",
        "seconrag_lite",
        "instructrag",
        "astuterag",
        "ours",
    ])
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-questions", type=int, default=500)
    parser.add_argument("--backend", default="transformers", choices=["fallback", "transformers", "vllm", "auto"])
    parser.add_argument("--model-path", default="/mnt/cpfs/chenxudu/workspace/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--disable-caf", action="store_true")
    parser.add_argument("--disable-nq-policy", action="store_true")
    parser.add_argument("--disable-doc-scorer", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-output", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_dir = config.get("data", {}).get("data_dir", "data_official_mixed_attack_nq500")
    attack_types = config.get("evaluation", {}).get("attack_types", DEFAULT_ATTACK_TYPES)
    rows = load_rows(data_dir, args.dataset, args.split, args.n_questions)
    attacks_by_type = {attack_type: load_attacks(data_dir, args.dataset, attack_type) for attack_type in attack_types}

    generator = build_generator(args)
    baseline_methods = build_baseline_methods(config) if args.method not in {"oracle", "ours"} else None
    orchestrator = build_orchestrator(config, generator, args) if args.method == "ours" else None

    trace_rows: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows, start=1):
        clean_docs = list(row.get("documents", []))
        if args.method == "oracle":
            eval_result = eval_oracle_case(generator, row, None, None)
        elif args.method == "ours":
            eval_result = eval_ours_case(orchestrator, row, clean_docs)
        else:
            eval_result = eval_baseline_case(args.method, baseline_methods, generator, row, clean_docs)
        trace_rows.append(
            make_trace_row(args.method, row, "clean", None, clean_docs, set(), eval_result, None)
        )

        for attack_type in attack_types:
            attack_row = find_attack_row(attacks_by_type[attack_type], row)
            injected = attack_docs_for(row, attack_row, attack_type)
            docs = clean_docs + injected
            attack_ids = {doc_id(doc, idx + len(clean_docs)) for idx, doc in enumerate(injected)}
            if args.method == "oracle":
                eval_result = eval_oracle_case(generator, row, attack_row, attack_type)
            elif args.method == "ours":
                eval_result = eval_ours_case(orchestrator, row, docs)
            else:
                eval_result = eval_baseline_case(args.method, baseline_methods, generator, row, docs)
            trace_rows.append(
                make_trace_row(
                    args.method,
                    row,
                    "attack",
                    attack_type,
                    docs,
                    attack_ids,
                    eval_result,
                    str(attack_row.get("target_answer", row.get("target_answer", ""))),
                )
            )
        if row_idx % 25 == 0:
            print(f"[Diagnose] method={args.method} processed={row_idx}/{len(rows)}", flush=True)

    summary = summarize_trace(trace_rows)
    summary.update(
        {
            "method": args.method,
            "config": args.config,
            "data_dir": data_dir,
            "dataset": args.dataset,
            "split": args.split,
            "n_questions": len(rows),
            "attack_types": attack_types,
            "backend": generator.backend,
            "disable_caf": args.disable_caf,
            "disable_nq_policy": args.disable_nq_policy,
            "disable_doc_scorer": args.disable_doc_scorer,
        }
    )
    trace_output = args.trace_output or str(Path(args.output).with_suffix(".trace.jsonl"))
    write_outputs(trace_rows, summary, args.output, trace_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
