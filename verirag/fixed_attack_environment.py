"""Fixed-attack PPO environment backed by aligned benchmark data."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .adversarial_doc_scorer import AdversarialDocScorer
from .learned_doc_scorer import LearnedAdversarialDocScorer
from .reward_function import RewardFunction, StepInfo
from .text_features import TextFeatureExtractor, get_doc_text


class FixedAttackRAGDefenseEnv:
    """
    PPO environment that samples real aligned queries and fixed attack files.

    Action space:
    - 0 KEEP_DOCS
    - 1 DROP_SUSPECT_DOCS
    - 2 RERANK_DOCS
    - 3 DROP_AND_RERANK
    - 4 ABSTAIN
    """

    def __init__(
        self,
        data_dir: str,
        datasets: List[str],
        reward_function: RewardFunction,
        generator: Optional[Any] = None,
        state_dim: int = 512,
        action_dim: int = 5,
        max_steps_per_episode: int = 1,
        attack_probability: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.datasets = datasets
        self.reward_function = reward_function
        self.generator = generator
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps_per_episode
        self.attack_probability = attack_probability
        self.config = config or {}

        self.max_docs = int(self.config.get("max_docs", 10))
        self.attack_type = self.config.get("attack_type", "poisonedrag")
        self.doc_filter_threshold = float(self.config.get("doc_filter_threshold", 0.18))
        self.max_doc_drop_fraction = float(self.config.get("max_doc_drop_fraction", 0.85))
        self.min_docs_after_filter = int(self.config.get("min_docs_after_filter", 1))

        feature_config = dict(self.config.get("feature_extractor", {}))
        feature_config.setdefault("max_docs", self.max_docs)
        self.text_features = TextFeatureExtractor(feature_config)
        scorer_config = {
            **self.config.get("doc_scorer", {}),
            "threshold": self.doc_filter_threshold,
            "max_drop_fraction": self.max_doc_drop_fraction,
            "min_docs": self.min_docs_after_filter,
            "use_source_prior": False,
        }
        heuristic_scorer = AdversarialDocScorer(scorer_config, feature_extractor=self.text_features)
        scorer_type = str(scorer_config.get("type", "heuristic")).lower()
        learned_path = scorer_config.get("model_path") or scorer_config.get("checkpoint")
        if scorer_type in {"learned", "ensemble"} or learned_path:
            scorer_config.setdefault("ensemble_weight", 1.0 if scorer_type == "learned" else 0.7)
            self.doc_scorer = LearnedAdversarialDocScorer(
                scorer_config,
                feature_extractor=self.text_features,
                heuristic_scorer=heuristic_scorer,
            )
        else:
            self.doc_scorer = heuristic_scorer

        self.samples = self._load_samples()
        self.attacks = self._load_attacks()
        self.attackable_samples = [
            sample for sample in self.samples
            if self._attack_row(sample) is not None
        ]
        if not self.samples:
            raise ValueError(f"No fixed-attack environment samples found in {self.data_dir}")
        if not self.attackable_samples:
            raise ValueError(f"No matching fixed attack rows found in {self.data_dir / 'attacks'}")

        self.current_sample: Optional[Dict[str, Any]] = None
        self.current_query: str = ""
        self.current_docs: List[Dict[str, Any]] = []
        self.ground_truth: List[str] = []
        self.target_answer: str = ""
        self.is_attacked = False
        self.step_count = 0
        self.history_actions: List[int] = []
        self.history_results: List[int] = []

        self.total_episodes = 0
        self.attack_success_count = 0
        self.correct_answer_count = 0
        self.detected_attack_count = 0

    def reset(self) -> Dict[str, torch.Tensor]:
        self.step_count = 0
        self.history_actions = []
        self.history_results = []
        self.total_episodes += 1

        use_attack = random.random() < self.attack_probability
        if use_attack:
            sample = random.choice(self.attackable_samples)
        else:
            sample = random.choice(self.samples)

        self.current_sample = sample
        self.current_query = sample.get("query") or sample.get("question") or ""
        self.ground_truth = self._answers(sample)
        self.current_docs = self._normalize_docs(sample.get("documents", []))
        self.target_answer = str(sample.get("target_answer") or "")
        self.is_attacked = False

        attack_row = self._attack_row(sample)
        if use_attack and attack_row:
            attack_docs = self._attack_docs(attack_row)
            if attack_docs:
                self.target_answer = str(attack_row.get("target_answer") or self.target_answer)
                self.current_docs = self._inject_attack_docs(self.current_docs, attack_docs)
                self.is_attacked = True

        self.current_docs = self.current_docs[: self.max_docs]
        return self._build_state()

    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        self.step_count += 1
        self.history_actions.append(int(action))

        filtered_docs, dropped_doc_ids, scored_docs = self._execute_action(int(action))
        if int(action) == 4:
            generated_answer = "I cannot answer this question because the retrieved evidence has high adversarial risk."
        else:
            generated_answer = self._generate_answer(self.current_query, filtered_docs)

        answer_correct = self._check_answer(generated_answer, self.ground_truth)
        attack_succeeded = bool(self.is_attacked and self.target_answer and self._check_answer(generated_answer, [self.target_answer]))
        attack_detected = int(action) == 4 or bool(dropped_doc_ids)

        if attack_succeeded:
            self.attack_success_count += 1
        if answer_correct:
            self.correct_answer_count += 1
        if self.is_attacked and attack_detected:
            self.detected_attack_count += 1

        result_code = 1 if answer_correct else 2 if attack_succeeded else 0
        self.history_results.append(result_code)

        step_info = StepInfo(
            is_attacked=self.is_attacked,
            attack_detected=attack_detected,
            attack_succeeded=attack_succeeded,
            answer_correct=answer_correct,
            verification_cost_ms=self._compute_verification_cost(int(action)),
            action_taken=int(action),
            false_positive=(attack_detected and not self.is_attacked),
            false_negative=(self.is_attacked and not attack_detected),
            final_step=True,
            ground_truth="; ".join(self.ground_truth),
            generated_answer=generated_answer,
            target_answer=self.target_answer,
        )
        reward_components = self.reward_function.compute(step_info=step_info, global_step=self.total_episodes)
        reward = reward_components.total

        info = {
            "is_attacked": self.is_attacked,
            "attack_detected": attack_detected,
            "attack_succeeded": attack_succeeded,
            "answer_correct": answer_correct,
            "generated_answer": generated_answer,
            "ground_truth": self.ground_truth,
            "target_answer": self.target_answer,
            "action": int(action),
            "action_name": self.action_name(int(action)),
            "dropped_doc_ids": dropped_doc_ids,
            "doc_scores": [score.to_dict() for score in scored_docs],
            "reward_components": {
                "correctness": reward_components.correctness,
                "safety": reward_components.safety,
                "efficiency": reward_components.efficiency,
                "verification": reward_components.verification,
            },
            "step_count": self.step_count,
        }

        return self._build_state(), reward, True, info

    def _build_state(self) -> Dict[str, torch.Tensor]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        docs = self.current_docs[: self.max_docs]
        doc_embeddings = self.text_features.doc_embeddings(docs)
        doc_scores = self.text_features.query_doc_scores(self.current_query, docs, doc_embeddings)
        n_docs = len(docs)

        doc_tensor = torch.zeros(1, n_docs, 768, dtype=torch.float32, device=device)
        if n_docs:
            emb = torch.from_numpy(doc_embeddings).to(device=device, dtype=torch.float32)
            if emb.shape[1] >= 768:
                doc_tensor[0] = emb[:, :768]
            else:
                doc_tensor[0, :, : emb.shape[1]] = emb

        if self.history_actions:
            actions = torch.tensor(self.history_actions[-10:], dtype=torch.long, device=device).unsqueeze(0)
            results = torch.tensor(self.history_results[-10:], dtype=torch.long, device=device).unsqueeze(0)
            history_mask = torch.ones_like(actions, dtype=torch.bool, device=device)
        else:
            actions = torch.zeros(1, 0, dtype=torch.long, device=device)
            results = torch.zeros(1, 0, dtype=torch.long, device=device)
            history_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)

        return {
            "query_tokens": self._query_tokens(self.current_query, device),
            "query_text": self.current_query,
            "doc_embeddings": doc_tensor,
            "doc_scores": torch.from_numpy(doc_scores).to(device=device, dtype=torch.float32).unsqueeze(0),
            "doc_masks": torch.ones(1, n_docs, dtype=torch.bool, device=device),
            "action_history": actions,
            "result_history": results,
            "history_mask": history_mask,
        }

    def _execute_action(self, action: int):
        docs = list(self.current_docs)
        scored_docs = self.doc_scorer.score(self.current_query, docs)
        dropped_doc_ids: List[str] = []
        if action == 0:
            return docs, dropped_doc_ids, scored_docs
        if action == 1:
            kept, scored_docs, dropped_doc_ids = self.doc_scorer.filter_docs(
                self.current_query,
                docs,
                threshold=self.doc_filter_threshold,
                min_docs=self.min_docs_after_filter,
                max_drop_fraction=self.max_doc_drop_fraction,
            )
            return kept, dropped_doc_ids, scored_docs
        if action == 2:
            return self._rerank(docs, scored_docs), dropped_doc_ids, scored_docs
        if action == 3:
            kept, scored_docs, dropped_doc_ids = self.doc_scorer.filter_docs(
                self.current_query,
                docs,
                threshold=self.doc_filter_threshold,
                min_docs=self.min_docs_after_filter,
                max_drop_fraction=self.max_doc_drop_fraction,
            )
            return self._rerank(kept, scored_docs), dropped_doc_ids, scored_docs
        if action == 4:
            return [], [], scored_docs
        return docs, dropped_doc_ids, scored_docs

    @staticmethod
    def _rerank(docs: List[Dict[str, Any]], scored_docs: List[Any]) -> List[Dict[str, Any]]:
        score_by_id = {score.doc_id: score for score in scored_docs}

        def key(doc: Dict[str, Any]):
            doc_id = str(doc.get("doc_id") or doc.get("id") or "")
            score = score_by_id.get(doc_id)
            if score is None:
                return (0.0, 0.0)
            return (score.attack_prob, -score.support_score)

        return sorted(docs, key=key)

    def _generate_answer(self, query: str, docs: List[Dict[str, Any]]) -> str:
        if self.generator is not None:
            try:
                return self.generator.generate(query, [get_doc_text(doc) for doc in docs])
            except Exception:
                pass

        remaining_attack = [
            doc for doc in docs
            if str(doc.get("source", "")).startswith("attack:")
        ]
        clean_docs = [doc for doc in docs if doc not in remaining_attack]
        if self.is_attacked and self.target_answer and len(remaining_attack) >= max(1, len(clean_docs)):
            return self.target_answer
        if self.ground_truth:
            return self.ground_truth[0]
        if docs:
            text = get_doc_text(docs[0])
            return text[:200]
        return ""

    def _load_samples(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for dataset in self.datasets:
            paths = [
                self.data_dir / f"{dataset}_train.jsonl",
                self.data_dir / f"{dataset}_validation.jsonl",
                self.data_dir / f"{dataset}_test.jsonl",
            ]
            for path in paths:
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                break
        return rows

    def _load_attacks(self) -> Dict[str, Dict[str, Any]]:
        attacks: Dict[str, Dict[str, Any]] = {}
        for dataset in self.datasets:
            path = self.data_dir / "attacks" / f"{dataset}_{self.attack_type}.jsonl"
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
                            attacks[str(key)] = row
        return attacks

    def _attack_row(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        metadata = sample.get("metadata", {}) or {}
        keys = [sample.get("id"), metadata.get("query_id")]
        for key in keys:
            if key and str(key) in self.attacks:
                return self.attacks[str(key)]
        return None

    @staticmethod
    def _attack_docs(row: Dict[str, Any]) -> List[str]:
        docs = row.get("poisoned_documents", [])
        texts = []
        if not isinstance(docs, list):
            return texts
        for doc in docs:
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or doc.get("document") or ""
            else:
                text = str(doc)
            text = text.strip()
            if text:
                texts.append(text)
        return texts

    @staticmethod
    def _normalize_docs(docs: Any) -> List[Dict[str, Any]]:
        if not isinstance(docs, list):
            return []
        out = []
        for idx, doc in enumerate(docs):
            if isinstance(doc, dict):
                normalized = dict(doc)
                normalized.setdefault("doc_id", normalized.get("id", f"doc_{idx}"))
                normalized.setdefault("text", get_doc_text(normalized))
                normalized.setdefault("source", normalized.get("source", "retrieved"))
                normalized.setdefault("metadata", normalized.get("metadata", {}) or {})
                normalized["metadata"].setdefault("rank", idx)
                out.append(normalized)
            else:
                out.append({"doc_id": f"doc_{idx}", "text": str(doc), "source": "retrieved", "metadata": {"rank": idx}})
        return out

    @staticmethod
    def _answers(sample: Dict[str, Any]) -> List[str]:
        answers = sample.get("answers")
        if isinstance(answers, list):
            return [str(answer) for answer in answers if str(answer).strip()]
        answer = sample.get("answer") or sample.get("ground_truth") or ""
        return [str(answer)] if str(answer).strip() else []

    def _inject_attack_docs(self, docs: List[Dict[str, Any]], attack_texts: List[str]) -> List[Dict[str, Any]]:
        out = list(docs)
        start_rank = len(out)
        for idx, text in enumerate(attack_texts):
            out.append({
                "doc_id": f"attack_{idx}",
                "text": text,
                "source": f"attack:{self.attack_type}",
                "metadata": {"rank": start_rank + idx, "fixed_attack": True},
            })
        return out

    @staticmethod
    def _normalize_answer(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\\b(a|an|the)\\b", " ", text)
        return " ".join(text.split())

    @classmethod
    def _check_answer(cls, generated: str, answers: Iterable[str]) -> bool:
        normalized_generated = cls._normalize_answer(str(generated))
        if not normalized_generated:
            return False
        for answer in answers:
            normalized_answer = cls._normalize_answer(str(answer))
            if normalized_answer and f" {normalized_answer} " in f" {normalized_generated} ":
                return True
        return False

    @staticmethod
    def _query_tokens(query: str, device: str) -> Dict[str, torch.Tensor]:
        max_length = 512
        ids = []
        for token in re.findall(r"[A-Za-z0-9]+", query.lower())[:max_length]:
            value = int.from_bytes(token.encode("utf-8")[:8].ljust(8, b"0"), "little")
            ids.append(value % 30521 + 1)
        ids = ids + [0] * (max_length - len(ids))
        mask = [1 if token_id else 0 for token_id in ids]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([mask], dtype=torch.long, device=device),
        }

    @staticmethod
    def _compute_verification_cost(action: int) -> float:
        return {0: 0.0, 1: 80.0, 2: 40.0, 3: 120.0, 4: 10.0}.get(action, 0.0)

    @staticmethod
    def action_name(action: int) -> str:
        return {
            0: "KEEP_DOCS",
            1: "DROP_SUSPECT_DOCS",
            2: "RERANK_DOCS",
            3: "DROP_AND_RERANK",
            4: "ABSTAIN",
        }.get(action, "UNKNOWN")

    def get_statistics(self) -> Dict[str, float]:
        return {
            "total_episodes": self.total_episodes,
            "attack_success_rate": self.attack_success_count / max(self.total_episodes, 1),
            "accuracy": self.correct_answer_count / max(self.total_episodes, 1),
            "attack_detection_rate": self.detected_attack_count / max(self.total_episodes, 1),
        }

    def seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
