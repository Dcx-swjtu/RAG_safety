"""Data helpers for JSONL corpora and document normalization."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a UTF-8 JSONL file, skipping blank lines."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write dictionaries to a UTF-8 JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_documents(documents: Sequence[Any]) -> List[Dict[str, Any]]:
    """Normalize strings or mixed document dictionaries to {doc_id, text, source} dicts."""
    normalized: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        if isinstance(doc, dict):
            item = dict(doc)
            item.setdefault("doc_id", item.get("id", f"doc_{idx}"))
            item.setdefault("text", item.get("content", item.get("document", "")))
            item.setdefault("source", item.get("metadata", {}).get("source", "unknown"))
        else:
            item = {
                "doc_id": f"doc_{idx}",
                "text": str(doc),
                "source": "unknown",
            }
        normalized.append(item)
    return normalized
