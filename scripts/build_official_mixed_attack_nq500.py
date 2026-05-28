#!/usr/bin/env python3
"""Build an NQ-500 mixed-attack benchmark from official attack artifacts.

The output keeps the VeriRAG fixed-attack schema:

  data_official_mixed_attack_nq500/
    nq_test.jsonl
    attacks/nq_<attack_type>.jsonl
    official_mixed_attack_manifest.json

The clean questions remain the project held-out NQ-500 split. Clean retrieval
contexts are rebuilt with the PoisonedRAG official Contriever top-k file when
available. Attack documents are adapted from official/public artifacts:

* PoisonedRAG LM-targeted results and official prefixing logic.
* GMTP released HotFlip and adversarial-decoding poisoned documents.
* RAGDefender released GARAG and Tan et al. artifacts.

Some public artifacts cover fewer than the held-out 500 queries. For those
rows, this builder fills deterministically from the same official artifact pool
and records the source mode in the manifest. This preserves a fixed 500-row
mixed benchmark while making coverage explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ATTACK_TYPES = [
    "poisonedrag_lm_targeted",
    "poisonedrag_hotflip",
    "garag",
    "tan_et_al",
    "advdecoding",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def norm_query(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def stable_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % n


def sample_query_id(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) or {}
    qid = metadata.get("query_id")
    if qid:
        return str(qid)
    sample_id = str(row.get("id", ""))
    if sample_id.startswith("nq_test_"):
        return sample_id[len("nq_test_") :]
    return sample_id


def doc_text(doc: Dict[str, Any]) -> str:
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("text") or doc.get("content") or doc.get("document") or "").strip()
    if title and text and not text.startswith(title):
        return f"{title}\n{text}"
    return text or title


def load_corpus(path: Path) -> Dict[str, Dict[str, Any]]:
    corpus: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("_id") or row.get("id") or "")
            if doc_id:
                corpus[doc_id] = row
    return corpus


def official_clean_docs(
    qid: str,
    fallback_docs: Sequence[Dict[str, Any]],
    retrieval: Dict[str, Dict[str, float]],
    corpus: Dict[str, Dict[str, Any]],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], str]:
    ranked = retrieval.get(qid) or {}
    docs: List[Dict[str, Any]] = []
    for rank, doc_id in enumerate(list(ranked.keys())[:top_k]):
        raw = corpus.get(str(doc_id))
        if not raw:
            continue
        text = doc_text(raw)
        if not text:
            continue
        docs.append(
            {
                "doc_id": str(doc_id),
                "title": raw.get("title", ""),
                "text": text,
                "source": "nq",
                "metadata": {
                    "rank": rank,
                    "selection_source": "poisonedrag_official_contriever_topk",
                    "retrieval_score": ranked.get(doc_id),
                },
            }
        )
    if docs:
        return docs, "poisonedrag_official_contriever_topk"

    normalized: List[Dict[str, Any]] = []
    for rank, doc in enumerate(fallback_docs):
        text = doc_text(doc)
        if not text:
            continue
        copied = dict(doc)
        copied["text"] = text
        copied.setdefault("doc_id", copied.get("id", f"doc_{rank}"))
        copied.setdefault("source", "nq")
        copied.setdefault("metadata", {})
        copied["metadata"] = dict(copied["metadata"] or {})
        copied["metadata"].setdefault("rank", rank)
        copied["metadata"]["selection_source"] = "heldout_qrels_fallback"
        normalized.append(copied)
    return normalized, "heldout_qrels_fallback"


def attack_doc(text: str, attack_type: str, idx: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": f"attack_{attack_type}_{idx}",
        "text": str(text).strip(),
        "source": f"official_attack:{attack_type}",
        "metadata": dict(metadata),
    }


def make_attack_row(
    sample: Dict[str, Any],
    attack_type: str,
    target_answer: str,
    poisoned_texts: Sequence[str],
    source_mode: str,
    source_repo: str,
    source_file: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    qid = sample_query_id(sample)
    docs = [
        attack_doc(
            text,
            attack_type,
            idx,
            {
                "rank": idx,
                "fixed_attack": True,
                "attack_label": 1,
                "source_mode": source_mode,
                "source_repo": source_repo,
                "source_file": source_file,
            },
        )
        for idx, text in enumerate(poisoned_texts)
        if str(text).strip()
    ]
    return {
        "id": f"{sample['id']}_{attack_type}",
        "sample_id": sample["id"],
        "query_id": qid,
        "dataset": "nq",
        "query": sample.get("query") or sample.get("question"),
        "attack_type": attack_type,
        "target_answer": target_answer,
        "poisoned_documents": docs,
        "metadata": {
            "official_attack_source": source_repo,
            "official_source_file": source_file,
            "source_mode": source_mode,
            **(extra or {}),
        },
    }


def first_answer(sample: Dict[str, Any]) -> str:
    answers = sample.get("answers")
    if isinstance(answers, list):
        for answer in answers:
            answer = str(answer).strip()
            if answer:
                return answer
    for key in ("answer", "ground_truth"):
        answer = str(sample.get(key) or "").strip()
        if answer:
            return answer
    return ""


def fallback_target(sample: Dict[str, Any]) -> str:
    existing = str(sample.get("target_answer") or "").strip()
    if existing and not existing.lower().startswith("not "):
        return existing
    answer = first_answer(sample)
    if not answer:
        return existing or "unsupported answer"
    return f"NOT {answer}"


def pool_record(pool: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    if not pool:
        return {}
    return pool[stable_index(key, len(pool))]


def record_texts(record: Dict[str, Any], keys: Sequence[str], max_docs: int) -> List[str]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            return [str(v).strip() for v in value[:max_docs] if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []


def official_query_map(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {norm_query(str(r.get("query") or r.get("question") or "")): r for r in records}


def build_poisonedrag_lm(
    samples: Sequence[Dict[str, Any]],
    poisonedrag_root: Path,
    adv_per_query: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    path = poisonedrag_root / "results" / "adv_targeted_results" / "nq.json"
    data = read_json(path)
    rows: List[Dict[str, Any]] = []
    stats = {"official_query_match": 0, "official_pool_fill": 0}
    pool = list(data.values())
    for sample in samples:
        qid = sample_query_id(sample)
        rec = data.get(qid)
        source_mode = "official_query_match"
        if not rec:
            rec = pool_record(pool, qid)
            source_mode = "official_pool_fill"
        stats[source_mode] += 1
        target = str(rec.get("incorrect answer") or fallback_target(sample))
        base_texts = list(rec.get("adv_texts", []))[:adv_per_query]
        query = str(sample.get("query") or sample.get("question") or "")
        poisoned = [f"{query}. {text}".strip() for text in base_texts]
        rows.append(
            make_attack_row(
                sample,
                "poisonedrag_lm_targeted",
                target,
                poisoned,
                source_mode,
                "PoisonedRAG",
                str(path),
                {"official_attack_method": "LM_targeted", "adv_per_query": adv_per_query},
            )
        )
    return rows, stats


def build_gmtp_attack(
    samples: Sequence[Dict[str, Any]],
    path: Path,
    attack_type: str,
    source_repo: str,
    text_keys: Sequence[str],
    adv_per_query: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records = read_json(path)
    by_qid = {str(r.get("query_id")): r for r in records if r.get("query_id")}
    by_query = official_query_map(records)
    stats = {"official_query_id_match": 0, "official_query_text_match": 0, "official_pool_fill": 0}
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        qid = sample_query_id(sample)
        rec = by_qid.get(qid)
        source_mode = "official_query_id_match"
        if not rec:
            rec = by_query.get(norm_query(str(sample.get("query") or sample.get("question") or "")))
            source_mode = "official_query_text_match"
        if not rec:
            rec = pool_record(records, qid)
            source_mode = "official_pool_fill"
        stats[source_mode] += 1
        target = str(rec.get("incorrect_answer") or fallback_target(sample))
        texts = record_texts(rec, text_keys, adv_per_query)
        rows.append(
            make_attack_row(
                sample,
                attack_type,
                target,
                texts,
                source_mode,
                source_repo,
                str(path),
                {"adv_per_query": adv_per_query},
            )
        )
    return rows, stats


def build_ragdefender_attack(
    samples: Sequence[Dict[str, Any]],
    path: Path,
    attack_type: str,
    source_repo: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records = read_json(path)
    by_query = official_query_map(records)
    stats = {"official_query_text_match": 0, "official_pool_fill": 0}
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        query = str(sample.get("query") or sample.get("question") or "")
        qid = sample_query_id(sample)
        rec = by_query.get(norm_query(query))
        source_mode = "official_query_text_match"
        if not rec:
            rec = pool_record(records, qid)
            source_mode = "official_pool_fill"
        stats[source_mode] += 1
        target = str(rec.get("incorrect_answer") or fallback_target(sample))
        text = str(rec.get("adversarial_document") or "").strip()
        rows.append(
            make_attack_row(
                sample,
                attack_type,
                target,
                [text],
                source_mode,
                source_repo,
                str(path),
            )
        )
    return rows, stats


def build_clean_rows(
    samples: Sequence[Dict[str, Any]],
    corpus_path: Path,
    retrieval_path: Path,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    corpus = load_corpus(corpus_path)
    retrieval = read_json(retrieval_path)
    rows: List[Dict[str, Any]] = []
    stats = {"poisonedrag_official_contriever_topk": 0, "heldout_qrels_fallback": 0}
    for sample in samples:
        qid = sample_query_id(sample)
        docs, source = official_clean_docs(qid, sample.get("documents", []), retrieval, corpus, top_k)
        stats[source] += 1
        row = dict(sample)
        row["documents"] = docs
        row["document"] = "\n\n".join(doc.get("text", "") for doc in docs)
        row["text"] = row["document"]
        row["metadata"] = dict(row.get("metadata", {}) or {})
        row["metadata"].update(
            {
                "query_id": qid,
                "clean_context_source": source,
                "official_retriever": "contriever",
                "official_top_k": top_k,
                "benchmark_family": "official_mixed_attack_nq500",
            }
        )
        rows.append(row)
    return rows, stats


def build(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to rebuild")
        shutil.rmtree(output)
    (output / "attacks").mkdir(parents=True, exist_ok=True)

    source_test = Path(args.source_test)
    samples = read_jsonl(source_test)
    if len(samples) < args.n_questions:
        raise ValueError(f"source test only has {len(samples)} rows, need {args.n_questions}")
    samples = samples[: args.n_questions]

    clean_rows, clean_stats = build_clean_rows(
        samples,
        corpus_path=Path(args.corpus),
        retrieval_path=Path(args.poisonedrag_root) / "results" / "beir_results" / "nq-contriever.json",
        top_k=args.top_k,
    )
    write_jsonl(output / "nq_test.jsonl", clean_rows)

    attacks: Dict[str, Dict[str, Any]] = {}

    rows, stats = build_poisonedrag_lm(samples, Path(args.poisonedrag_root), args.adv_per_query)
    write_jsonl(output / "attacks" / "nq_poisonedrag_lm_targeted.jsonl", rows)
    attacks["poisonedrag_lm_targeted"] = {"count": len(rows), "source_stats": stats}

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
    write_jsonl(output / "attacks" / "nq_poisonedrag_hotflip.jsonl", rows)
    attacks["poisonedrag_hotflip"] = {"count": len(rows), "source_stats": stats}

    rows, stats = build_ragdefender_attack(
        samples,
        Path(args.ragdefender_root) / "artifacts" / "GARAG" / "garag_nq.json",
        "garag",
        "RAGDefender/GARAG",
    )
    write_jsonl(output / "attacks" / "nq_garag.jsonl", rows)
    attacks["garag"] = {"count": len(rows), "source_stats": stats}

    rows, stats = build_ragdefender_attack(
        samples,
        Path(args.ragdefender_root) / "artifacts" / "tan" / "tan_nq.json",
        "tan_et_al",
        "RAGDefender/Tan-et-al",
    )
    write_jsonl(output / "attacks" / "nq_tan_et_al.jsonl", rows)
    attacks["tan_et_al"] = {"count": len(rows), "source_stats": stats}

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
    write_jsonl(output / "attacks" / "nq_advdecoding.jsonl", rows)
    attacks["advdecoding"] = {"count": len(rows), "source_stats": stats}

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": "data_official_mixed_attack_nq500",
        "description": (
            "Held-out NQ-500 with official Contriever clean contexts and mixed "
            "official/public attack artifacts converted to VeriRAG fixed-attack JSONL."
        ),
        "dataset": "nq",
        "split": "test",
        "num_questions": len(samples),
        "clean_source": str(source_test),
        "clean_contexts": {
            "top_k": args.top_k,
            "retriever": "contriever",
            "source_file": str(Path(args.poisonedrag_root) / "results" / "beir_results" / "nq-contriever.json"),
            "stats": clean_stats,
        },
        "attack_types": ATTACK_TYPES,
        "attacks": attacks,
        "settings": {
            "top_k": args.top_k,
            "adv_per_query": args.adv_per_query,
            "n_questions": args.n_questions,
            "official_artifact_fill_policy": (
                "query-id/query-text match when available; deterministic same-artifact pool fill "
                "for official artifacts whose public release covers fewer than 500 held-out queries"
            ),
        },
        "schema": {
            "clean": "nq_test.jsonl",
            "attacks": "attacks/nq_<attack_type>.jsonl",
            "attack_row_keys": ["id", "sample_id", "query_id", "target_answer", "poisoned_documents"],
        },
    }
    with (output / "official_mixed_attack_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    rag_root = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-test", default=str(root / "data_official_benchmark_500" / "nq_test.jsonl"))
    parser.add_argument("--output", default=str(root / "data_official_mixed_attack_nq500"))
    parser.add_argument("--corpus", default=str(rag_root / "data" / "nq" / "corpus.jsonl"))
    parser.add_argument("--poisonedrag-root", default=str(rag_root / "data_process" / "PoisonedRAG"))
    parser.add_argument("--ragdefender-root", default=str(rag_root / "data_process" / "RAGDefender"))
    parser.add_argument("--gmtp-root", default=str(rag_root / "data_process" / "GMTP"))
    parser.add_argument("--n-questions", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--adv-per-query", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
