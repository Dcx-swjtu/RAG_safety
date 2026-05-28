"""
Run a single VeriRAG defense pass from local documents.

Examples:
    python scripts/run_defense.py \
        --query "What is the revenue of Company X?" \
        --docs examples/docs.jsonl

The script works without Qwen weights: it uses deterministic generator and
lexical retriever fallbacks. Provide --model-path and --backend vllm/transformers
when a local Qwen checkpoint is available.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verirag.claim_extractor import ClaimExtractor
from verirag.cross_validator import CrossValidator
from verirag.defense_orchestrator import DefenseOrchestrator
from verirag.generator import QwenGenerator
from verirag.policy_network import VerificationPolicyNetwork
from verirag.retriever import DenseRetriever
from verirag.utils.data_utils import load_jsonl, normalize_documents


def load_documents(path: str) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        rows = load_jsonl(str(input_path))
    elif input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        rows = loaded if isinstance(loaded, list) else loaded.get("documents", [])
    else:
        with input_path.open("r", encoding="utf-8") as f:
            rows = [{"doc_id": f"doc_{i}", "text": line.strip()} for i, line in enumerate(f) if line.strip()]
    return normalize_documents(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VeriRAG defense for one query.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config YAML path.")
    parser.add_argument("--query", required=True, help="User query.")
    parser.add_argument("--docs", required=True, help="JSON/JSONL/TXT document file.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved docs.")
    parser.add_argument("--model-path", default="./models/Qwen-8B-Chat", help="Local Qwen checkpoint path.")
    parser.add_argument("--backend", default="fallback", choices=["auto", "vllm", "transformers", "fallback"])
    parser.add_argument(
        "--allow-remote-model-download",
        action="store_true",
        help="Allow HuggingFace downloads for the policy query encoder.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = dict(config.get("model", {}))
    model_config["allow_remote_model_download"] = args.allow_remote_model_download
    if not args.allow_remote_model_download:
        model_config.setdefault("use_pretrained_encoder", False)

    documents = load_documents(args.docs)
    retriever = DenseRetriever(backend="lexical")
    retriever.build_index(documents)
    retrieved = retriever.search(args.query, top_k=args.top_k)
    retrieved_docs = [
        {
            "doc_id": item.doc_id,
            "text": item.text,
            "source": item.metadata.get("source", "unknown"),
            "score": item.score,
        }
        for item in retrieved
    ]

    generator = QwenGenerator(
        model_path=args.model_path,
        backend=args.backend,
        load_model=args.backend != "fallback",
    )
    policy_network = VerificationPolicyNetwork(config=model_config)
    defense = DefenseOrchestrator(
        policy_network=policy_network,
        claim_extractor=ClaimExtractor(config=config.get("claim_extractor", {})),
        cross_validator=CrossValidator(config=config.get("cross_validator", {})),
        base_llm=generator,
        config=config.get("defense", {}),
    )

    result = defense.defend(args.query, retrieved_docs)
    print(json.dumps({
        "query": result.query,
        "answer": result.final_answer,
        "status": result.status.value,
        "confidence": result.confidence,
        "policy_action": result.policy_action,
        "risk_indicators": result.risk_indicators,
        "detected_attacks": result.detected_attacks,
        "layers": {
            "source": result.source_layer.__dict__ if result.source_layer else None,
            "evidence": result.evidence_layer.__dict__ if result.evidence_layer else None,
            "claim": result.claim_layer.__dict__ if result.claim_layer else None,
            "answer": result.answer_layer.__dict__ if result.answer_layer else None,
        },
        "retrieved_docs": retrieved_docs,
        "trace": result.execution_trace,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
