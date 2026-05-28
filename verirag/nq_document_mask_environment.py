"""NQ-only document mask environment with verification reward signals."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .adversarial_doc_scorer import AdversarialDocScorer
from .generator import GenerationConfig, QwenGenerator
from .learned_doc_scorer import LearnedAdversarialDocScorer
from .text_features import TextFeatureExtractor, get_doc_text
from .nq_doc_features import NQDocFeatureBuilder


def _normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _contains_answer(text: str, answers: Sequence[str]) -> bool:
    normalized_text = f" {_normalize_answer(text)} "
    if not normalized_text.strip():
        return False
    for answer in answers:
        norm_answer = _normalize_answer(answer)
        if norm_answer and f" {norm_answer} " in normalized_text:
            return True
    return False


class NQDocumentMaskEnv:
    """
    Fixed-attack NQ training environment for per-document evidence selection.

    The policy acts directly on each document through a binary keep/drop mask and
    an optional global abstain action.
    """

    def __init__(
        self,
        data_dir: str,
        attack_types: Optional[Sequence[str]] = None,
        split: str = "test",
        max_docs: int = 10,
        attack_probability: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
        generator: Optional[Any] = None,
    ):
        self.data_dir = Path(data_dir)
        self.attack_types = list(attack_types or ["poisonedrag"])
        self.split = split
        self.max_docs = int(max_docs)
        self.attack_probability = float(attack_probability)
        self.config = config or {}
        self.qwen_reward_cfg = dict(self.config.get("qwen_reward", {}))
        self.qwen_reward_enabled = bool(self.qwen_reward_cfg.get("enabled", False))
        self.qwen_reward_sample_rate = float(self.qwen_reward_cfg.get("sample_rate", 1.0))
        self.qwen_reward_cache_enabled = bool(self.qwen_reward_cfg.get("cache", True))
        self.qwen_reward_cache_size = int(self.qwen_reward_cfg.get("cache_size", 4096))
        self.qwen_reward_cache: Dict[str, Dict[str, Any]] = {}
        self.qwen_generator = generator
        if self.qwen_reward_enabled and self.qwen_generator is None:
            self.qwen_generator = QwenGenerator(
                model_path=str(self.qwen_reward_cfg.get("model_path", "./models/Qwen-8B-Chat")),
                backend=str(self.qwen_reward_cfg.get("backend", "fallback")),
                tensor_parallel_size=int(self.qwen_reward_cfg.get("tensor_parallel_size", 1)),
                gpu_memory_utilization=float(self.qwen_reward_cfg.get("gpu_memory_utilization", 0.8)),
                temperature=float(self.qwen_reward_cfg.get("temperature", 0.1)),
                top_p=float(self.qwen_reward_cfg.get("top_p", 0.9)),
                max_tokens=int(self.qwen_reward_cfg.get("max_tokens", 96)),
                load_model=bool(self.qwen_reward_cfg.get("load_model", True)),
                device=self.qwen_reward_cfg.get("device"),
            )
        self.feature_extractor = TextFeatureExtractor(
            {
                **self.config.get("feature_extractor", {}),
                "max_docs": self.max_docs,
            }
        )

        scorer_cfg = dict(self.config.get("doc_scorer", {}))
        scorer_cfg.setdefault("use_source_prior", False)
        heuristic = AdversarialDocScorer(scorer_cfg, feature_extractor=self.feature_extractor)
        scorer_type = str(scorer_cfg.get("type", "heuristic")).lower()
        if scorer_type in {"learned", "ensemble"} or scorer_cfg.get("model_path"):
            scorer_cfg.setdefault("ensemble_weight", 0.75 if scorer_type == "ensemble" else 1.0)
            self.doc_scorer = LearnedAdversarialDocScorer(
                scorer_cfg,
                feature_extractor=self.feature_extractor,
                heuristic_scorer=heuristic,
            )
        else:
            self.doc_scorer = heuristic
        self.doc_feature_builder = NQDocFeatureBuilder(self.feature_extractor, self.doc_scorer)

        self.samples = self._load_samples()
        self.attacks = self._load_attacks()
        self.attackable_samples = [
            sample for sample in self.samples
            if any(self._attack_row(sample, attack_type) for attack_type in self.attack_types)
        ]
        if not self.samples:
            raise ValueError(f"No NQ samples found in {self.data_dir}")
        if not self.attackable_samples:
            raise ValueError(f"No NQ attack rows found in {self.data_dir / 'attacks'}")

        self.current_sample: Optional[Dict[str, Any]] = None
        self.current_query = ""
        self.current_answers: List[str] = []
        self.current_docs: List[Dict[str, Any]] = []
        self.current_labels: List[int] = []
        self.current_support_labels: List[int] = []
        self.current_attack_type = "clean"
        self.current_target_answer = ""
        self.is_attacked = False
        self.total_episodes = 0

    @property
    def feature_dim(self) -> int:
        return self.doc_feature_builder.feature_dim

    def reset(self) -> Dict[str, torch.Tensor]:
        self.total_episodes += 1
        use_attack = random.random() < self.attack_probability
        if use_attack:
            sample = random.choice(self.attackable_samples)
            available = [
                attack_type for attack_type in self.attack_types
                if self._attack_row(sample, attack_type) is not None
            ]
            attack_type = random.choice(available)
        else:
            sample = random.choice(self.samples)
            attack_type = "clean"

        self.current_sample = sample
        self.current_query = sample.get("query") or sample.get("question") or ""
        self.current_answers = self._answers(sample)
        clean_docs = self._normalize_docs(sample.get("documents", []), label=0)
        docs = list(clean_docs)
        labels = [0 for _ in clean_docs]
        support_labels = [1 if _contains_answer(get_doc_text(doc), self.current_answers) else 0 for doc in clean_docs]
        self.current_target_answer = str(sample.get("target_answer") or "")
        self.current_attack_type = attack_type
        self.is_attacked = False

        if use_attack and attack_type != "clean":
            row = self._attack_row(sample, attack_type)
            attack_docs = self._attack_docs(row or {}, attack_type, start_rank=len(docs))
            if attack_docs:
                self.current_target_answer = str((row or {}).get("target_answer") or self.current_target_answer)
                docs.extend(attack_docs)
                labels.extend([1 for _ in attack_docs])
                support_labels.extend([0 for _ in attack_docs])
                self.is_attacked = True

        packed = list(zip(docs, labels, support_labels))
        packed = packed[: self.max_docs]
        self.current_docs = [row[0] for row in packed]
        self.current_labels = [row[1] for row in packed]
        self.current_support_labels = [row[2] for row in packed]
        return self._build_state()

    def step(self, action: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        keep_mask = action["keep_mask"].detach().float().view(-1).cpu().numpy()
        abstain = bool(float(action.get("abstain", torch.tensor([0.0])).view(-1)[0].detach().cpu()) >= 0.5)
        n_docs = len(self.current_docs)
        keep_mask = keep_mask[:n_docs]
        if keep_mask.shape[0] < n_docs:
            keep_mask = np.pad(keep_mask, (0, n_docs - keep_mask.shape[0]))
        keep_bool = keep_mask >= 0.5

        labels = np.asarray(self.current_labels, dtype=np.int64)
        support_labels = np.asarray(self.current_support_labels, dtype=np.int64)
        attack_docs_total = int(labels.sum())
        attack_docs_kept = int((labels * keep_bool).sum())
        attack_docs_dropped = attack_docs_total - attack_docs_kept
        clean_total = int((1 - labels).sum())
        clean_kept = int(((1 - labels) * keep_bool).sum())
        support_total = int(support_labels.sum())
        support_kept = int((support_labels * keep_bool).sum())

        target_leakage = bool(self.is_attacked and not abstain and attack_docs_kept > 0)
        contradiction = bool(self.is_attacked and attack_docs_kept > 0 and clean_kept > 0)
        support_retained = support_kept / max(support_total, 1)
        clean_retained = clean_kept / max(clean_total, 1)
        attack_removed = attack_docs_dropped / max(attack_docs_total, 1)
        false_positive = bool((not self.is_attacked) and (abstain or clean_retained < 0.5))
        attack_detected = bool(self.is_attacked and (abstain or attack_docs_dropped > 0))
        attack_succeeded = bool(self.is_attacked and target_leakage and attack_docs_kept >= max(clean_kept, 1))
        answer_correct = bool((not abstain) and support_kept > 0 and not attack_succeeded)
        qwen_feedback = self._qwen_reward_feedback(keep_bool, abstain)
        if qwen_feedback.get("enabled") and not qwen_feedback.get("skipped"):
            answer_correct = bool(qwen_feedback.get("answer_correct", answer_correct))
            attack_succeeded = bool(qwen_feedback.get("attack_succeeded", attack_succeeded))

        reward = self._reward(
            answer_correct=answer_correct,
            attack_succeeded=attack_succeeded,
            attack_detected=attack_detected,
            false_positive=false_positive,
            abstain=abstain,
            support_retained=support_retained,
            clean_retained=clean_retained,
            attack_removed=attack_removed,
            target_leakage=target_leakage,
            contradiction=contradiction,
        )
        reward += float(qwen_feedback.get("reward_delta", 0.0))
        info = {
            "query": self.current_query,
            "is_attacked": self.is_attacked,
            "attack_type": self.current_attack_type,
            "attack_detected": attack_detected,
            "attack_succeeded": attack_succeeded,
            "answer_correct": answer_correct,
            "false_positive": false_positive,
            "abstain": abstain,
            "num_docs": n_docs,
            "kept_docs": int(keep_bool.sum()),
            "attack_docs_total": attack_docs_total,
            "attack_docs_kept": attack_docs_kept,
            "clean_docs_kept": clean_kept,
            "verify_signals": {
                "support_retained": support_retained,
                "clean_retained": clean_retained,
                "attack_removed": attack_removed,
                "target_leakage": float(target_leakage),
                "contradiction": float(contradiction),
            },
            "qwen_reward": qwen_feedback,
        }
        return self._build_state(), reward, True, info

    def _qwen_reward_feedback(self, keep_bool: np.ndarray, abstain: bool) -> Dict[str, Any]:
        if (
            not self.qwen_reward_enabled
            or self.qwen_generator is None
            or random.random() > self.qwen_reward_sample_rate
        ):
            return {"enabled": False}

        kept_docs = [
            self.current_docs[idx]
            for idx in range(min(len(self.current_docs), len(keep_bool)))
            if bool(keep_bool[idx])
        ]
        if abstain or not kept_docs:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "abstain" if abstain else "no_kept_docs",
                "generated_answer": "",
                "answer_correct": False,
                "attack_succeeded": False,
                "reward_delta": 0.0,
            }

        max_docs = int(self.qwen_reward_cfg.get("max_docs", 5))
        max_doc_chars = int(self.qwen_reward_cfg.get("max_doc_chars", 900))
        doc_texts = [get_doc_text(doc)[:max_doc_chars] for doc in kept_docs[:max_docs]]
        doc_ids = [str(doc.get("doc_id", idx)) for idx, doc in enumerate(kept_docs[:max_docs])]
        cache_key = json.dumps(
            {
                "query": self.current_query,
                "doc_ids": doc_ids,
                "target": self.current_target_answer,
                "answers": self.current_answers,
            },
            sort_keys=True,
        )
        if self.qwen_reward_cache_enabled and cache_key in self.qwen_reward_cache:
            return dict(self.qwen_reward_cache[cache_key])

        gen_config = GenerationConfig(
            temperature=float(self.qwen_reward_cfg.get("generation_temperature", 0.0)),
            top_p=float(self.qwen_reward_cfg.get("generation_top_p", 0.9)),
            max_tokens=int(self.qwen_reward_cfg.get("generation_max_tokens", 64)),
        )
        try:
            generated_answer = str(self.qwen_generator.generate_answer(self.current_query, doc_texts, gen_config))
        except TypeError:
            generated_answer = str(self.qwen_generator.generate_answer(self.current_query, doc_texts))

        answer_correct = _contains_answer(generated_answer, self.current_answers)
        target_answers = [self.current_target_answer] if self.current_target_answer else []
        attack_succeeded = bool(self.is_attacked and _contains_answer(generated_answer, target_answers))
        verification: Dict[str, Any] = {}
        if bool(self.qwen_reward_cfg.get("use_verifier", True)):
            try:
                verification = dict(self.qwen_generator.verify_answer(self.current_query, generated_answer, doc_texts))
            except AttributeError:
                verification = {}

        reward_delta = self._qwen_reward_delta(answer_correct, attack_succeeded, verification)
        feedback = {
            "enabled": True,
            "skipped": False,
            "generated_answer": generated_answer,
            "answer_correct": bool(answer_correct),
            "attack_succeeded": bool(attack_succeeded),
            "verification": verification,
            "reward_delta": float(reward_delta),
            "doc_ids": doc_ids,
        }
        if self.qwen_reward_cache_enabled:
            if len(self.qwen_reward_cache) >= self.qwen_reward_cache_size:
                self.qwen_reward_cache.pop(next(iter(self.qwen_reward_cache)))
            self.qwen_reward_cache[cache_key] = dict(feedback)
        return feedback

    def _qwen_reward_delta(
        self,
        answer_correct: bool,
        attack_succeeded: bool,
        verification: Dict[str, Any],
    ) -> float:
        cfg = self.qwen_reward_cfg
        reward = 0.0
        if answer_correct:
            reward += float(cfg.get("correct_bonus", 4.0))
        elif not self.is_attacked:
            reward -= float(cfg.get("clean_wrong_penalty", 4.0))
        if attack_succeeded:
            reward -= float(cfg.get("target_answer_penalty", 10.0))

        if verification:
            confidence = float(verification.get("confidence", 0.0) or 0.0)
            supported = bool(verification.get("is_correct", False))
            if answer_correct and supported:
                reward += float(cfg.get("verified_correct_bonus", 2.0)) * confidence
            elif not supported:
                reward -= float(cfg.get("unsupported_penalty", 2.0)) * max(0.0, 1.0 - confidence)
        return reward

    def _reward(
        self,
        answer_correct: bool,
        attack_succeeded: bool,
        attack_detected: bool,
        false_positive: bool,
        abstain: bool,
        support_retained: float,
        clean_retained: float,
        attack_removed: float,
        target_leakage: bool,
        contradiction: bool,
    ) -> float:
        reward = 0.0
        if self.is_attacked:
            reward += 8.0 * attack_removed
            reward += 4.0 * support_retained
            if attack_detected and not attack_succeeded:
                reward += 6.0
            if answer_correct:
                reward += 5.0
            if attack_succeeded:
                reward -= 12.0
            if target_leakage:
                reward -= 5.0
            if contradiction:
                reward -= 2.0
            if abstain:
                reward -= 1.5
        else:
            if answer_correct:
                reward += 8.0
            reward += 3.0 * clean_retained
            if false_positive:
                reward -= 10.0
            if abstain:
                reward -= 6.0

        kept_fraction_penalty = 0.3 * max(0.0, clean_retained - 1.0)
        return float(reward - kept_fraction_penalty)

    def _build_state(self) -> Dict[str, torch.Tensor]:
        docs = self.current_docs
        n_docs = len(docs)
        features = np.zeros((self.max_docs, self.feature_dim), dtype=np.float32)
        doc_mask = np.zeros((self.max_docs,), dtype=np.bool_)
        labels = np.zeros((self.max_docs,), dtype=np.float32)
        support_labels = np.zeros((self.max_docs,), dtype=np.float32)
        if n_docs:
            features[:n_docs] = self._doc_features(self.current_query, docs)
            doc_mask[:n_docs] = True
            labels[:n_docs] = np.asarray(self.current_labels, dtype=np.float32)
            support_labels[:n_docs] = np.asarray(self.current_support_labels, dtype=np.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "doc_features": torch.from_numpy(features).unsqueeze(0).to(device=device),
            "doc_mask": torch.from_numpy(doc_mask).unsqueeze(0).to(device=device),
            "attack_labels": torch.from_numpy(labels).unsqueeze(0).to(device=device),
            "support_labels": torch.from_numpy(support_labels).unsqueeze(0).to(device=device),
        }

    def _doc_features(self, query: str, docs: List[Dict[str, Any]]) -> np.ndarray:
        return self.doc_feature_builder.build(query, docs)

    def _load_samples(self) -> List[Dict[str, Any]]:
        path = self.data_dir / f"nq_{self.split}.jsonl"
        if not path.exists():
            for fallback in ("train", "validation", "test"):
                path = self.data_dir / f"nq_{fallback}.jsonl"
                if path.exists():
                    break
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = row.get("metadata", {}) or {}
                if metadata.get("eval_gold") is False:
                    continue
                rows.append(row)
        return rows

    def _load_attacks(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        attacks: Dict[str, Dict[str, Dict[str, Any]]] = {attack_type: {} for attack_type in self.attack_types}
        for attack_type in self.attack_types:
            path = self.data_dir / "attacks" / self.split / f"nq_{attack_type}.jsonl"
            if not path.exists():
                path = self.data_dir / "attacks" / f"nq_{attack_type}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for key in (row.get("sample_id"), row.get("query_id"), row.get("id")):
                        if key:
                            attacks[attack_type][str(key)] = row
        return attacks

    def _attack_row(self, sample: Dict[str, Any], attack_type: str) -> Optional[Dict[str, Any]]:
        metadata = sample.get("metadata", {}) or {}
        for key in (sample.get("id"), metadata.get("query_id")):
            if key and str(key) in self.attacks.get(attack_type, {}):
                return self.attacks[attack_type][str(key)]
        return None

    @staticmethod
    def _normalize_docs(docs: Any, label: int = 0) -> List[Dict[str, Any]]:
        if not isinstance(docs, list):
            return []
        out: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            if isinstance(doc, dict):
                normalized = dict(doc)
                normalized.setdefault("doc_id", normalized.get("id", f"doc_{idx}"))
                normalized.setdefault("text", get_doc_text(normalized))
                normalized.setdefault("source", normalized.get("source", "nq"))
                normalized.setdefault("metadata", normalized.get("metadata", {}) or {})
                normalized["metadata"].setdefault("rank", idx)
            else:
                normalized = {"doc_id": f"doc_{idx}", "text": str(doc), "source": "nq", "metadata": {"rank": idx}}
            normalized["metadata"]["attack_label"] = label
            out.append(normalized)
        return out

    @staticmethod
    def _attack_docs(row: Dict[str, Any], attack_type: str, start_rank: int) -> List[Dict[str, Any]]:
        docs = row.get("poisoned_documents", [])
        out: List[Dict[str, Any]] = []
        if not isinstance(docs, list):
            return out
        for idx, doc in enumerate(docs):
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or doc.get("document") or ""
            else:
                text = str(doc)
            text = text.strip()
            if not text:
                continue
            out.append(
                {
                    "doc_id": f"attack_{attack_type}_{idx}",
                    "text": text,
                    "source": f"attack:{attack_type}",
                    "metadata": {"rank": start_rank + idx, "fixed_attack": True, "attack_label": 1},
                }
            )
        return out

    @staticmethod
    def _answers(sample: Dict[str, Any]) -> List[str]:
        answers = sample.get("answers")
        if isinstance(answers, list):
            return [str(answer) for answer in answers if str(answer).strip()]
        answer = sample.get("answer") or sample.get("ground_truth") or ""
        return [str(answer)] if str(answer).strip() else []

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
