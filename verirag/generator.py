"""
Qwen generator integration for VeriRAG.

The class provides one interface for all LLM-backed roles described in the
development guide:
- answer generation
- few-shot claim extraction
- few-shot cross validation
- answer verification

Heavy inference backends are optional. If vLLM/transformers or a local Qwen
checkpoint is unavailable, the class falls back to deterministic local logic so
the rest of the pipeline, tests, and data-construction scripts remain runnable.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class GenerationConfig:
    """Generation parameters shared by vLLM and transformers backends."""

    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512
    stop: List[str] = field(default_factory=list)


class QwenGenerator:
    """
    Qwen wrapper with graceful local fallback.

    Args:
        model_path: Local model directory or HuggingFace model id.
        backend: "auto", "vllm", "transformers", or "fallback".
        load_model: Set False to force local deterministic behavior.
    """

    CLAIM_EXTRACTION_PROMPT = """Extract factual claims from the given document.
For each claim, identify subject, predicate, object, value, and type.
Return only a JSON array.

Document:
{document}
"""

    CROSS_VALIDATION_PROMPT = """You are validating factual consistency.
Return JSON with verdict, confidence, conflicts, and risk_score.

Claims:
{claims_text}
"""

    ANSWER_VERIFICATION_PROMPT = """Verify whether the answer is supported by the documents.
Return JSON with is_correct, confidence, issues, and supported_facts.

Question: {question}
Answer: {answer}
Documents:
{documents}
"""

    def __init__(
        self,
        model_path: str = "./models/Qwen-8B-Chat",
        backend: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 512,
        load_model: bool = True,
        trust_remote_code: bool = True,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.backend = "fallback"
        self.llm = None
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_vision_language_model = False
        self.device = device
        self.default_generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        if load_model and backend in {"auto", "vllm"}:
            try:
                from vllm import LLM

                self.llm = LLM(
                    model=model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    trust_remote_code=trust_remote_code,
                    dtype="float16",
                )
                self.tokenizer = self.llm.get_tokenizer()
                self.backend = "vllm"
                return
            except Exception as exc:
                if backend == "vllm":
                    raise RuntimeError(f"Failed to load vLLM backend: {exc}") from exc

        if load_model and backend in {"auto", "transformers"}:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

                local_only = os.path.isdir(model_path)
                model_info = self._read_model_info(model_path)
                model_type = str(model_info.get("model_type", ""))
                architectures = model_info.get("architectures") or []
                self.is_vision_language_model = (
                    "vl" in model_type or
                    any("VL" in str(arch) or "Vision" in str(arch) for arch in architectures)
                )

                if self.is_vision_language_model:
                    from transformers import Qwen3VLForConditionalGeneration

                    self.processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=trust_remote_code,
                        local_files_only=local_only,
                    )
                    self.tokenizer = getattr(self.processor, "tokenizer", None)
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path,
                        trust_remote_code=trust_remote_code,
                        dtype="auto",
                        device_map="auto" if torch.cuda.is_available() else None,
                        local_files_only=local_only,
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=trust_remote_code,
                        local_files_only=local_only,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=trust_remote_code,
                        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        local_files_only=local_only,
                    )
                self.model.eval()
                self.backend = "transformers"
                return
            except Exception as exc:
                if backend == "transformers":
                    raise RuntimeError(f"Failed to load transformers backend: {exc}") from exc

    def generate_answer(
        self,
        query: str,
        documents: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate a RAG answer from query and source documents."""
        config = config or self.default_generation_config
        context = "\n\n".join(f"Document {i + 1}: {doc}" for i, doc in enumerate(documents))
        prompt = self._chat_prompt(
            system="You answer questions using only the provided documents.",
            user=f"Documents:\n{context}\n\nQuestion: {query}\nAnswer:",
        )
        if self.backend != "fallback":
            return self._generate_text(prompt, config).strip()
        return self._fallback_answer(query, documents)

    def generate(self, query: str, context: str) -> str:
        """Compatibility method used by DefenseOrchestrator base_llm integration."""
        docs = [part.strip() for part in context.split("\n\n") if part.strip()]
        if not docs and context.strip():
            docs = [context.strip()]
        return self.generate_answer(query, docs)

    def generate_answer_batch(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Batch answer generation."""
        config = config or self.default_generation_config
        if self.backend == "vllm":
            prompts = [
                self._chat_prompt(
                    system="You answer questions using only the provided documents.",
                    user=(
                        "Documents:\n"
                        + "\n\n".join(f"Document {i + 1}: {doc}" for i, doc in enumerate(docs))
                        + f"\n\nQuestion: {query}\nAnswer:"
                    ),
                )
                for query, docs in zip(queries, documents_list)
            ]
            outputs = self._generate_text_batch(prompts, config)
            return [text.strip() for text in outputs]
        return [self.generate_answer(query, docs, config) for query, docs in zip(queries, documents_list)]

    def extract_claims(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Extract structured claims from documents."""
        if self.backend != "fallback":
            all_claims: List[Dict[str, Any]] = []
            for doc_idx, doc in enumerate(documents):
                prompt = self._chat_prompt(
                    system="Extract factual claims and output only valid JSON.",
                    user=self.CLAIM_EXTRACTION_PROMPT.format(document=doc[:2000]),
                )
                raw = self._generate_text(prompt, GenerationConfig(temperature=0.1, max_tokens=1024))
                all_claims.extend(self._parse_claim_json(raw, doc_idx))
            return all_claims

        from .claim_extractor import ClaimExtractor

        extractor = ClaimExtractor({"rule_engine_enabled": True, "llm_extractor_enabled": False})
        claims = extractor.extract(documents, [f"doc_{i}" for i in range(len(documents))])
        return [
            {
                "subject": claim.subject,
                "predicate": claim.predicate,
                "object": claim.object,
                "value": claim.value if claim.value is not None else claim.object,
                "type": claim.claim_type.value.upper(),
                "doc_id": claim.doc_id,
                "confidence": claim.confidence,
            }
            for claim in claims
        ]

    def cross_validate_claims(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-validate claims using Qwen when available, deterministic logic otherwise."""
        if self.backend != "fallback":
            claims_text = "\n".join(
                f"- [{c.get('type', 'FACTUAL')}] {c.get('subject', '')} "
                f"{c.get('predicate', '')} {c.get('value', c.get('object', ''))} "
                f"(from {c.get('doc_id', 'unknown')})"
                for c in claims
            )
            prompt = self._chat_prompt(
                system="You validate factual consistency across sources.",
                user=self.CROSS_VALIDATION_PROMPT.format(claims_text=claims_text),
            )
            raw = self._generate_text(prompt, GenerationConfig(temperature=0.1, max_tokens=512))
            return self._parse_validation_json(raw)

        conflicts: List[Dict[str, Any]] = []
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for claim in claims:
            key = (
                str(claim.get("subject", "")).strip().lower(),
                str(claim.get("predicate", "")).strip().lower(),
                str(claim.get("type", "FACTUAL")).strip().lower(),
            )
            groups.setdefault(key, []).append(claim)

        for group in groups.values():
            if len(group) < 2:
                continue
            values = [str(c.get("value", c.get("object", ""))).strip() for c in group]
            if len(set(values)) <= 1:
                continue
            nums = [self._extract_number(v) for v in values]
            if all(v is not None for v in nums):
                numeric_values = [float(v) for v in nums if v is not None]
                denom = max(max(abs(v) for v in numeric_values), 1e-8)
                diff = (max(numeric_values) - min(numeric_values)) / denom
                if diff > 0.05:
                    conflicts.append(
                        {
                            "type": "VALUE_MISMATCH",
                            "claims": values,
                            "reason": f"numeric values differ by {diff:.3f}",
                        }
                    )
            else:
                conflicts.append(
                    {
                        "type": "ENTITY_OR_TEXT_MISMATCH",
                        "claims": values,
                        "reason": "claim values differ across sources",
                    }
                )

        risk_score = min(1.0, 0.35 * len(conflicts))
        verdict = "CONFLICT" if conflicts else "CONSISTENT"
        return {
            "verdict": verdict,
            "confidence": 1.0 - risk_score if conflicts else 0.95,
            "conflicts": conflicts,
            "risk_score": risk_score,
        }

    def verify_answer(self, question: str, answer: str, documents: List[str]) -> Dict[str, Any]:
        """Verify whether an answer is supported by source documents."""
        if self.backend != "fallback":
            docs_text = "\n\n".join(f"Document {i + 1}: {doc[:700]}" for i, doc in enumerate(documents))
            prompt = self._chat_prompt(
                system="Verify answer correctness against source documents. Output JSON only.",
                user=self.ANSWER_VERIFICATION_PROMPT.format(
                    question=question,
                    answer=answer,
                    documents=docs_text,
                ),
            )
            raw = self._generate_text(prompt, GenerationConfig(temperature=0.1, max_tokens=512))
            return self._parse_verification_json(raw)

        doc_text = " ".join(documents).lower()
        answer_terms = [t for t in re.findall(r"\w+", answer.lower()) if len(t) > 2]
        if not answer_terms:
            return {"is_correct": False, "confidence": 0.0, "issues": ["empty answer"], "supported_facts": []}
        supported = [term for term in answer_terms if term in doc_text]
        support_ratio = len(supported) / max(len(set(answer_terms)), 1)
        return {
            "is_correct": support_ratio >= 0.35,
            "confidence": min(0.95, support_ratio),
            "issues": [] if support_ratio >= 0.35 else ["low lexical support"],
            "supported_facts": supported[:10],
        }

    def _chat_prompt(self, system: str, user: str) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    def _generate_text(self, prompt: str, config: GenerationConfig) -> str:
        if self.backend == "vllm":
            return self._generate_text_batch([prompt], config)[0]

        if self.backend == "transformers":
            import torch

            if self.processor is not None and self.is_vision_language_model:
                inputs = self.processor(text=[prompt], return_tensors="pt")
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=config.temperature > 0,
                    temperature=max(config.temperature, 1e-6),
                    top_p=config.top_p,
                    max_new_tokens=config.max_tokens,
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            generated = output_ids[0, inputs["input_ids"].shape[1] :]
            decoder = self.processor if self.processor is not None else self.tokenizer
            if hasattr(decoder, "batch_decode"):
                return decoder.batch_decode(
                    [generated],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            return self.tokenizer.decode(generated, skip_special_tokens=True)

        return ""

    def _generate_text_batch(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop or ["<|im_end|>"],
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    @staticmethod
    def _read_model_info(model_path: str) -> Dict[str, Any]:
        if not os.path.isdir(model_path):
            return {}
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _parse_claim_json(raw: str, doc_idx: int) -> List[Dict[str, Any]]:
        parsed = QwenGenerator._extract_json(raw, expected=list)
        claims: List[Dict[str, Any]] = []
        if not isinstance(parsed, list):
            return claims
        for item in parsed:
            if not isinstance(item, dict):
                continue
            claims.append(
                {
                    "subject": str(item.get("subject", "")),
                    "predicate": str(item.get("predicate", "")),
                    "object": str(item.get("object", "")),
                    "value": str(item.get("value", item.get("object", ""))),
                    "type": str(item.get("type", "FACTUAL")).upper(),
                    "doc_id": item.get("doc_id", f"doc_{doc_idx}"),
                }
            )
        return claims

    @staticmethod
    def _parse_validation_json(raw: str) -> Dict[str, Any]:
        parsed = QwenGenerator._extract_json(raw, expected=dict)
        if isinstance(parsed, dict):
            parsed.setdefault("verdict", "UNCERTAIN")
            parsed.setdefault("confidence", 0.5)
            parsed.setdefault("conflicts", [])
            parsed.setdefault("risk_score", 0.5 if parsed["verdict"] == "UNCERTAIN" else 0.0)
            return parsed
        return {"verdict": "UNCERTAIN", "confidence": 0.5, "conflicts": [], "risk_score": 0.5}

    @staticmethod
    def _parse_verification_json(raw: str) -> Dict[str, Any]:
        parsed = QwenGenerator._extract_json(raw, expected=dict)
        if isinstance(parsed, dict):
            parsed.setdefault("is_correct", False)
            parsed.setdefault("confidence", 0.0)
            parsed.setdefault("issues", [])
            parsed.setdefault("supported_facts", [])
            return parsed
        return {"is_correct": False, "confidence": 0.0, "issues": ["parse error"], "supported_facts": []}

    @staticmethod
    def _extract_json(raw: str, expected: type) -> Any:
        raw = raw.strip()
        candidates: Iterable[str]
        if expected is list:
            candidates = re.findall(r"\[[\s\S]*\]", raw)
        else:
            candidates = re.findall(r"\{[\s\S]*\}", raw)
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, expected):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        match = re.search(r"[\$€£¥]?\s*([\d,]+(?:\.\d+)?)\s*([KMBTkmbt]?)", text)
        if not match:
            return None
        multiplier = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
        suffix = match.group(2).upper()
        return float(match.group(1).replace(",", "")) * multiplier.get(suffix, 1.0)

    @staticmethod
    def _fallback_answer(query: str, documents: List[str]) -> str:
        if not documents:
            return "I do not have enough retrieved evidence to answer this question."

        query_terms = {term for term in re.findall(r"\w+", query.lower()) if len(term) > 2}
        best_doc = max(
            documents,
            key=lambda doc: len(query_terms & set(re.findall(r"\w+", doc.lower()))),
        )
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", best_doc) if s.strip()]
        if not sentences:
            return best_doc[:500]
        return sentences[0][:500]
