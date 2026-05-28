"""
Module 5: Defense Orchestrator（防御编排器）

功能:
- 协调所有模块执行
- 四层验证架构（L1-L4）
- 与Policy Network交互
- 输出最终答案+验证报告

执行流程:
1. 声明提取 (Claim Extractor)
2. 交叉验证 -> conflict_indicators
3. Policy Network决策 -> action
4. 执行验证动作
5. 四层最终验证
6. 输出答案+验证报告
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto

import torch
import torch.nn as nn
import numpy as np

from .claim_extractor import ClaimExtractor, Claim
from .cross_validator import CrossValidator, ValidationReport, ConsistencyLabel
from .policy_network import VerificationPolicyNetwork
from .adversarial_doc_scorer import AdversarialDocScorer
from .learned_doc_scorer import LearnedAdversarialDocScorer
from .text_features import TextFeatureExtractor
from .nq_doc_features import NQDocFeatureBuilder
from .nq_doc_policy import NQDocumentActionPolicy
from .conflict_aware_generation import ConflictAwareEvidenceController


# ==================== 数据类型定义 ====================

class FinalAnswerStatus(Enum):
    """最终答案状态"""
    VERIFIED = "verified"       # 验证通过
    REJECTED = "rejected"       # 拒绝回答
    UNCERTAIN = "uncertain"     # 不确定
    PARTIAL = "partial"         # 部分回答


@dataclass
class LayerResult:
    """单层验证结果"""
    layer_name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0


@dataclass
class DefenseResult:
    """
    防御结果

    包含:
    - 最终答案
    - 验证报告
    - 置信度评分
    - 审计日志
    """
    query: str = ""
    final_answer: str = ""
    status: FinalAnswerStatus = FinalAnswerStatus.VERIFIED
    confidence: float = 0.0

    # 四层验证结果
    source_layer: Optional[LayerResult] = None
    evidence_layer: Optional[LayerResult] = None
    claim_layer: Optional[LayerResult] = None
    answer_layer: Optional[LayerResult] = None

    # 策略决策
    policy_action: str = ""
    policy_confidence: float = 0.0

    # 攻击检测
    detected_attacks: List[Dict[str, Any]] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    doc_scores: List[Dict[str, Any]] = field(default_factory=list)
    filtered_doc_ids: List[str] = field(default_factory=list)

    # 审计
    execution_trace: List[str] = field(default_factory=list)
    module_timings: Dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0


# ==================== Defense Orchestrator ====================

class DefenseOrchestrator:
    """
    防御编排器: VeriRAG的主入口

    负责:
    - 协调各模块执行
    - 四层验证
    - 输出结构化结果
    """

    def __init__(
        self,
        policy_network: VerificationPolicyNetwork,
        claim_extractor: ClaimExtractor,
        cross_validator: CrossValidator,
        base_llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化防御编排器

        Args:
            policy_network: 训练好的策略网络
            claim_extractor: 声明提取器
            cross_validator: 交叉验证器
            base_llm: 基础LLM（用于答案生成）
            config: 配置字典
        """
        self.policy_network = policy_network
        self.claim_extractor = claim_extractor
        self.cross_validator = cross_validator
        self.base_llm = base_llm
        self.config = config or {}

        # 性能参数
        self.light_verify_threshold = self.config.get('light_verify_threshold', 0.3)
        self.deep_verify_threshold = self.config.get('deep_verify_threshold', 0.6)
        self.reject_threshold = self.config.get('reject_threshold', 0.8)
        self.policy_deterministic = self.config.get('policy_deterministic', True)

        # 真实特征 + 文档级攻击评分。默认不使用 source=attack:* 标签泄漏。
        feature_config = dict(self.config.get('feature_extractor', {}))
        feature_config.setdefault('max_docs', self.policy_network.config.get('max_docs', 10))
        self.text_features = TextFeatureExtractor(feature_config)

        scorer_config = dict(self.config.get('doc_scorer', {}))
        self.enable_doc_scorer = self.config.get('enable_doc_scorer', True)
        self.doc_filter_threshold = float(
            self.config.get('doc_filter_threshold', scorer_config.get('threshold', 0.18))
        )
        self.doc_score_detection_threshold = float(
            self.config.get('doc_score_detection_threshold', self.doc_filter_threshold)
        )
        self.min_docs_after_filter = int(self.config.get('min_docs_after_filter', scorer_config.get('min_docs', 1)))
        self.max_doc_drop_fraction = float(
            self.config.get('max_doc_drop_fraction', scorer_config.get('max_drop_fraction', 0.85))
        )
        scorer_config.setdefault('threshold', self.doc_filter_threshold)
        scorer_config.setdefault('min_docs', self.min_docs_after_filter)
        scorer_config.setdefault('max_drop_fraction', self.max_doc_drop_fraction)
        scorer_config.setdefault('use_source_prior', False)
        scorer_type = str(scorer_config.get('type', 'heuristic')).lower()
        learned_path = scorer_config.get('model_path') or scorer_config.get('checkpoint')
        heuristic_scorer = AdversarialDocScorer(scorer_config, feature_extractor=self.text_features)
        if scorer_type in {'learned', 'ensemble'} or learned_path:
            scorer_config.setdefault('ensemble_weight', 1.0 if scorer_type == 'learned' else 0.7)
            self.doc_scorer = LearnedAdversarialDocScorer(
                scorer_config,
                feature_extractor=self.text_features,
                heuristic_scorer=heuristic_scorer,
            )
        else:
            self.doc_scorer = heuristic_scorer

        self.enable_nq_doc_policy = bool(self.config.get('enable_nq_doc_policy', False))
        self.nq_doc_policy_deterministic = bool(self.config.get('nq_doc_policy_deterministic', True))
        self.nq_doc_policy_max_docs = int(self.config.get('nq_doc_policy_max_docs', 32))
        self.nq_doc_policy = None
        self.nq_doc_feature_builder = NQDocFeatureBuilder(self.text_features, self.doc_scorer)
        if self.enable_nq_doc_policy:
            checkpoint_path = self.config.get('nq_doc_policy_checkpoint')
            if not checkpoint_path:
                raise ValueError('enable_nq_doc_policy=true requires nq_doc_policy_checkpoint')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            policy_config = dict(checkpoint.get('config', {}))
            policy_config.setdefault('input_dim', self.nq_doc_feature_builder.feature_dim)
            self.nq_doc_policy = NQDocumentActionPolicy(**policy_config)
            state_dict = checkpoint.get('model_state', checkpoint.get('state_dict'))
            if state_dict is None:
                raise KeyError('NQ doc policy checkpoint must contain model_state or state_dict')
            self.nq_doc_policy.load_state_dict(state_dict)
            policy_device = next(self.policy_network.parameters()).device
            self.nq_doc_policy.to(policy_device)
            self.nq_doc_policy.eval()

        self.enable_conflict_aware_generation = bool(
            self.config.get('enable_conflict_aware_generation', True)
        )
        self.enable_verification_guided_evidence = bool(
            self.config.get('enable_verification_guided_evidence', True)
        )
        self.conflict_aware_query_instruction = bool(
            self.config.get('conflict_aware_query_instruction', False)
        )
        self.conflict_aware_controller = ConflictAwareEvidenceController(
            self.config.get('conflict_aware_generation', {})
        )

        # 统计
        self.total_queries = 0
        self.rejected_queries = 0
        self.verified_queries = 0

    def defend(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        """
        主防御入口

        完整防御流程:
        1. 声明提取
        2. 交叉验证
        3. Policy Network决策
        4. 执行验证动作
        5. 四层最终验证
        6. 输出结果

        Args:
            query: 用户查询
            retrieved_docs: 检索到的文档列表

        Returns:
            DefenseResult包含最终答案和验证报告
        """
        start_time = time.time()
        self.total_queries += 1

        result = DefenseResult(query=query)
        execution_trace = []
        module_timings = {}

        try:
            # ========== Step 1: 声明提取 ==========
            t0 = time.time()
            extraction_result = self._step_claim_extraction(query, retrieved_docs)
            claims = extraction_result.get('claims', [])
            module_timings['claim_extraction'] = (time.time() - t0) * 1000
            execution_trace.append(f"Extracted {len(claims)} claims from {len(retrieved_docs)} docs")

            # ========== Step 2: 交叉验证 ==========
            t0 = time.time()
            validation_reports = self._step_cross_validation(claims)
            conflict_indicators = self._extract_conflict_indicators(validation_reports)
            module_timings['cross_validation'] = (time.time() - t0) * 1000
            execution_trace.append(
                f"Validation: {len([r for r in validation_reports if r.label == ConsistencyLabel.CONFLICT])} conflicts found"
            )

            # ========== Step 2.5: 文档级攻击评分/Policy过滤 ==========
            t0 = time.time()
            policy_input_docs = retrieved_docs
            if self.enable_nq_doc_policy:
                policy_input_docs, doc_score_rows, dropped_doc_ids, doc_policy_info = self._apply_nq_doc_policy(
                    query, retrieved_docs
                )
                result.doc_scores = doc_score_rows
                result.filtered_doc_ids = dropped_doc_ids
                if dropped_doc_ids:
                    result.risk_indicators.append(
                        f"NQ document policy dropped {len(dropped_doc_ids)} docs"
                    )
                    dropped_set = set(dropped_doc_ids)
                    result.detected_attacks.extend([
                        {
                            'type': 'nq_doc_policy_drop',
                            'doc_id': row.get('doc_id'),
                            'severity': 'high' if row.get('attack_prob', 0.0) >= 0.65 else 'medium',
                            'attack_prob': row.get('attack_prob', 0.0),
                            'keep_prob': row.get('nq_doc_policy_keep_prob', 0.0),
                            'reasons': row.get('reasons', []),
                        }
                        for row in doc_score_rows
                        if row.get('doc_id') in dropped_set
                    ])
                execution_trace.append(
                    f"NQ doc policy: kept {len(policy_input_docs)}/{len(retrieved_docs)} docs "
                    f"(abstain={doc_policy_info.get('abstain', False)})"
                )
                if doc_policy_info.get('abstain', False):
                    result.status = FinalAnswerStatus.REJECTED
                    result.final_answer = "I cannot answer this question because the retrieved evidence has high adversarial risk."
                    result.confidence = float(doc_policy_info.get('abstain_prob', 0.9))
                    result.policy_action = 'NQ_DOC_POLICY_ABSTAIN'
                    result.policy_confidence = result.confidence
                    self.rejected_queries += 1
                    module_timings['doc_scoring'] = (time.time() - t0) * 1000
                    result.execution_trace = execution_trace
                    result.module_timings = module_timings
                    result.total_time_ms = (time.time() - start_time) * 1000
                    result.source_layer = LayerResult("L1_Source", True, {}, 0.0)
                    result.evidence_layer = LayerResult("L2_Evidence", False, {'reason': 'nq_doc_policy_abstain'}, 1.0)
                    result.claim_layer = LayerResult("L3_Claim", True, {}, conflict_indicators.get('overall_risk', 0.0))
                    result.answer_layer = LayerResult("L4_Answer", False, {}, 1.0)
                    return result
            elif self.enable_doc_scorer:
                conflict_doc_ids = self._extract_conflict_doc_ids(validation_reports)
                policy_input_docs, doc_scores, dropped_doc_ids = self.doc_scorer.filter_docs(
                    query,
                    retrieved_docs,
                    conflict_doc_ids=conflict_doc_ids,
                    threshold=self.doc_filter_threshold,
                    min_docs=self.min_docs_after_filter,
                    max_drop_fraction=self.max_doc_drop_fraction,
                )
                dropped_set_for_scores = set(dropped_doc_ids)
                result.doc_scores = [score.to_dict() for score in doc_scores]
                for row in result.doc_scores:
                    row['doc_scorer_kept'] = str(row.get('doc_id')) not in dropped_set_for_scores
                result.filtered_doc_ids = dropped_doc_ids
                high_risk_docs = [
                    score for score in doc_scores
                    if score.attack_prob >= self.doc_score_detection_threshold
                ]
                if dropped_doc_ids:
                    result.risk_indicators.append(
                        f"Adversarial document scorer dropped {len(dropped_doc_ids)} docs"
                    )
                    dropped_set = set(dropped_doc_ids)
                    result.detected_attacks.extend([
                        {
                            'type': 'adversarial_document',
                            'doc_id': score.doc_id,
                            'severity': 'high' if score.attack_prob >= 0.65 else 'medium',
                            'attack_prob': score.attack_prob,
                            'reasons': score.reasons,
                        }
                        for score in high_risk_docs
                        if score.doc_id in dropped_set
                    ])
                execution_trace.append(
                    f"Doc scorer: dropped {len(dropped_doc_ids)}/{len(retrieved_docs)} docs "
                    f"(threshold={self.doc_filter_threshold:.2f})"
                )
            module_timings['doc_scoring'] = (time.time() - t0) * 1000

            # ========== Step 3: Policy Network决策 ==========
            t0 = time.time()
            policy_decision = self._step_policy_decision(
                query, policy_input_docs, conflict_indicators
            )
            action = policy_decision['action']
            action_name = policy_decision['action_name']
            result.policy_action = action_name
            result.policy_confidence = policy_decision.get('confidence', 0.0)
            module_timings['policy_decision'] = (time.time() - t0) * 1000
            execution_trace.append(f"Policy decision: {action_name} (confidence={result.policy_confidence:.3f})")

            # ========== Step 4: 执行验证动作 ==========
            t0 = time.time()
            if action == 0:  # KEEP_DOCS
                filtered_docs = policy_input_docs
                execution_trace.append("Kept scored documents")
            elif action == 1:  # DROP_SUSPECT_DOCS
                filtered_docs = policy_input_docs if self.enable_doc_scorer else self._execute_light_verify(policy_input_docs, claims)
                execution_trace.append("Dropped suspected documents")
            elif action == 2:  # RERANK_DOCS
                filtered_docs = self._rerank_docs_by_attack_score(policy_input_docs, result.doc_scores)
                execution_trace.append("Reranked documents by adversarial score")
            elif action == 3:  # DROP_AND_RERANK
                filtered_docs = self._rerank_docs_by_attack_score(policy_input_docs, result.doc_scores)
                execution_trace.append("Dropped suspected documents and reranked remaining evidence")
            elif action == 4:  # ABSTAIN
                result.status = FinalAnswerStatus.REJECTED
                result.final_answer = "I cannot answer this question because the retrieved evidence has high adversarial risk."
                result.confidence = policy_decision.get('confidence', 0.9)
                self.rejected_queries += 1
                execution_trace.append("Abstained due to high document/query risk")
                module_timings['action_execution'] = (time.time() - t0) * 1000

                # 填充审计信息
                result.execution_trace = execution_trace
                result.module_timings = module_timings
                result.total_time_ms = (time.time() - start_time) * 1000

                # 填充四层验证（基础版）
                result.source_layer = LayerResult("L1_Source", True, {}, 0.0)
                result.evidence_layer = LayerResult("L2_Evidence", True, {}, 0.0)
                result.claim_layer = LayerResult("L3_Claim", True, {}, conflict_indicators.get('overall_risk', 0.0))
                result.answer_layer = LayerResult("L4_Answer", False, {}, 0.0)

                return result

            module_timings['action_execution'] = (time.time() - t0) * 1000

            # ========== Step 4.5: 冲突感知证据控制 ==========
            t0 = time.time()
            generation_query = query
            if self.enable_conflict_aware_generation:
                controller_input_docs = (
                    retrieved_docs if self.enable_verification_guided_evidence and result.doc_scores else filtered_docs
                )
                evidence_result = self.conflict_aware_controller.filter_evidence(
                    controller_input_docs,
                    doc_scores=result.doc_scores,
                    validation_reports=validation_reports,
                )
                if evidence_result.rescued_doc_ids:
                    rescued = set(evidence_result.rescued_doc_ids)
                    result.filtered_doc_ids = [
                        doc_id for doc_id in result.filtered_doc_ids if doc_id not in rescued
                    ]
                    result.detected_attacks = [
                        row for row in result.detected_attacks
                        if str(row.get('doc_id')) not in rescued
                    ]
                    result.risk_indicators.append(
                        f"Verification-guided evidence rescued {len(evidence_result.rescued_doc_ids)} support docs"
                    )
                if evidence_result.dropped_doc_ids:
                    existing = set(result.filtered_doc_ids)
                    rescued = set(evidence_result.rescued_doc_ids)
                    for doc_id in evidence_result.dropped_doc_ids:
                        if doc_id not in existing and doc_id not in rescued:
                            result.filtered_doc_ids.append(doc_id)
                    result.risk_indicators.append(
                        f"Conflict-aware generation dropped {len(evidence_result.dropped_doc_ids)} docs"
                    )
                    result.detected_attacks.extend([
                        {
                            'type': 'conflict_aware_evidence_drop',
                            'doc_id': doc_id,
                            'severity': 'high',
                        }
                        for doc_id in evidence_result.dropped_doc_ids
                        if doc_id not in set(evidence_result.rescued_doc_ids)
                    ])
                execution_trace.append(
                    f"Verification-guided evidence: kept {len(evidence_result.docs)}/{len(controller_input_docs)} docs "
                    f"(risk={evidence_result.risk_score:.3f}, support={evidence_result.support_score:.3f}, "
                    f"abstain={evidence_result.should_abstain})"
                )
                if evidence_result.notes:
                    result.risk_indicators.extend(evidence_result.notes)
                if evidence_result.should_abstain:
                    result.status = FinalAnswerStatus.REJECTED
                    result.final_answer = "I cannot answer this question because the remaining evidence is conflicting or unsafe."
                    result.confidence = max(0.5, evidence_result.risk_score)
                    result.policy_action = 'CONFLICT_AWARE_ABSTAIN'
                    result.policy_confidence = result.confidence
                    self.rejected_queries += 1
                    module_timings['conflict_aware_generation'] = (time.time() - t0) * 1000
                    result.execution_trace = execution_trace
                    result.module_timings = module_timings
                    result.total_time_ms = (time.time() - start_time) * 1000
                    result.source_layer = LayerResult("L1_Source", True, {}, 0.0)
                    result.evidence_layer = LayerResult(
                        "L2_Evidence",
                        False,
                        {'reason': evidence_result.abstain_reason or 'conflict_aware_abstain'},
                        evidence_result.risk_score,
                    )
                    result.claim_layer = LayerResult("L3_Claim", True, {}, conflict_indicators.get('overall_risk', 0.0))
                    result.answer_layer = LayerResult("L4_Answer", False, {}, evidence_result.risk_score)
                    return result
                filtered_docs = evidence_result.docs
                if self.conflict_aware_query_instruction:
                    generation_query = self.conflict_aware_controller.generation_instruction(query)
            module_timings['conflict_aware_generation'] = (time.time() - t0) * 1000

            # ========== Step 5: 答案生成 ==========
            t0 = time.time()
            answer = self._generate_answer(generation_query, filtered_docs)
            module_timings['answer_generation'] = (time.time() - t0) * 1000
            execution_trace.append(f"Generated answer: {answer[:100]}...")

            # ========== Step 6: 四层最终验证 ==========
            t0 = time.time()
            layer_results = self._four_layer_verify(
                query, filtered_docs, claims, validation_reports, answer
            )
            module_timings['four_layer_verify'] = (time.time() - t0) * 1000

            result.source_layer = layer_results.get('source')
            result.evidence_layer = layer_results.get('evidence')
            result.claim_layer = layer_results.get('claim')
            result.answer_layer = layer_results.get('answer')

            # 判断验证结果
            all_passed = all(
                lr.passed for lr in [result.source_layer, result.evidence_layer,
                                     result.claim_layer, result.answer_layer]
                if lr is not None
            )

            residual_risk = conflict_indicators.get('has_conflict', False) or \
                conflict_indicators.get('overall_risk', 0.0) >= self.deep_verify_threshold

            if all_passed and not residual_risk:
                result.status = FinalAnswerStatus.VERIFIED
                result.confidence = 0.85
            else:
                result.status = FinalAnswerStatus.UNCERTAIN
                result.confidence = max(0.1, 0.5 * (1.0 - conflict_indicators.get('overall_risk', 0.0)))

            result.final_answer = answer
            self.verified_queries += 1

            # 攻击检测指标
            if conflict_indicators.get('has_conflict', False):
                result.risk_indicators.append("Cross-document conflict detected")
            if conflict_indicators.get('precision_drift', False):
                result.detected_attacks.append({
                    'type': 'precision_drift',
                    'severity': 'high',
                })

        except Exception as e:
            execution_trace.append(f"Error during defense: {str(e)}")
            result.status = FinalAnswerStatus.UNCERTAIN
            result.final_answer = f"Error: {str(e)}"
            result.confidence = 0.0

        # 填充审计信息
        result.execution_trace = execution_trace
        result.module_timings = module_timings
        result.total_time_ms = (time.time() - start_time) * 1000

        return result

    def _step_claim_extraction(
        self, query: str, docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Step 1: 声明提取"""
        doc_texts = [d.get('text', d.get('content', '')) for d in docs]
        doc_ids = [d.get('doc_id', d.get('id', f'doc_{i}')) for i, d in enumerate(docs)]

        claims = self.claim_extractor.extract(doc_texts, doc_ids)

        return {
            'claims': claims,
            'num_docs': len(docs),
            'num_claims': len(claims),
        }

    def _step_cross_validation(
        self, claims: List[Claim]
    ) -> List[ValidationReport]:
        """Step 2: 交叉验证"""
        if not claims:
            return []

        # 按组验证
        group_results = self.cross_validator.validate_all_groups(claims)

        return list(group_results.values())

    def _extract_conflict_indicators(
        self, reports: List[ValidationReport]
    ) -> Dict[str, Any]:
        """提取冲突指标（用于Policy Network决策）"""
        if not reports:
            return {
                'has_conflict': False,
                'conflict_ratio': 0.0,
                'overall_risk': 0.0,
                'precision_drift': False,
                'semantic_inversion': False,
            }

        n_conflict = sum(1 for r in reports if r.label == ConsistencyLabel.CONFLICT)
        n_total = len(reports)

        precision_drift = any(
            any(c.conflict_type.value == 'precision_drift' for c in r.conflicts)
            for r in reports
        )
        semantic_inversion = any(
            any(c.conflict_type.value == 'semantic_inversion' for c in r.conflicts)
            for r in reports
        )

        overall_risk = max((r.risk_score for r in reports), default=0.0)

        return {
            'has_conflict': n_conflict > 0,
            'conflict_ratio': n_conflict / max(n_total, 1),
            'overall_risk': overall_risk,
            'precision_drift': precision_drift,
            'semantic_inversion': semantic_inversion,
            'num_groups': n_total,
            'num_conflicts': n_conflict,
        }

    def _extract_conflict_doc_ids(self, reports: List[ValidationReport]) -> List[str]:
        """Return document ids that participated in cross-validation conflicts."""
        conflict_doc_ids = set()
        for report in reports:
            if report.label != ConsistencyLabel.CONFLICT:
                continue
            for claim in getattr(report, 'source_claims', []) or []:
                doc_id = getattr(claim, 'doc_id', None)
                if doc_id:
                    conflict_doc_ids.add(str(doc_id))
        return sorted(conflict_doc_ids)

    def _step_policy_decision(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        conflict_indicators: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Step 3: Policy Network决策"""
        # High-confidence symbolic signals should override an untrained policy.
        # The neural policy still handles low/medium-risk cases.
        overall_risk = float(conflict_indicators.get('overall_risk', 0.0))
        if overall_risk >= self.reject_threshold:
            return {
                'action': 4,
                'action_name': 'ABSTAIN',
                'confidence': overall_risk,
            }
        if conflict_indicators.get('has_conflict', False) or overall_risk >= self.deep_verify_threshold:
            return {
                'action': 3,
                'action_name': 'DROP_AND_RERANK',
                'confidence': max(overall_risk, 0.6),
            }

        use_neural_policy = self.config.get('use_neural_policy', False)
        if not use_neural_policy:
            if len(docs) <= 1:
                return {
                    'action': 0,
                    'action_name': 'KEEP_DOCS',
                    'confidence': max(0.5, 1.0 - overall_risk),
                }
            if overall_risk >= self.light_verify_threshold:
                return {
                    'action': 1,
                    'action_name': 'DROP_SUSPECT_DOCS',
                    'confidence': max(0.6, 1.0 - overall_risk),
                }
            return {
                'action': 0,
                'action_name': 'KEEP_DOCS',
                'confidence': 1.0 - overall_risk,
            }

        # 构建Policy Network输入：使用真实 query/doc 特征，不再随机生成 doc embedding。
        device = next(self.policy_network.parameters()).device

        # History（空）
        action_history = torch.zeros(1, 0, dtype=torch.long, device=device)
        result_history = torch.zeros(1, 0, dtype=torch.long, device=device)

        policy_inputs = self.text_features.build_policy_inputs(
            query=query,
            docs=docs,
            policy_network=self.policy_network,
            device=device,
            action_history=action_history,
            result_history=result_history,
        )

        # Policy决策
        with torch.no_grad():
            decision = self.policy_network.select_action(
                policy_inputs, deterministic=self.policy_deterministic
            )

        return {
            'action': decision['action'],
            'action_name': decision['action_name'],
            'confidence': decision.get('value', 0.0),
        }


    def _apply_nq_doc_policy(
        self,
        query: str,
        docs: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
        """Apply the learned NQ document-level keep/drop mask."""
        if self.nq_doc_policy is None or not docs:
            return docs, [], [], {'abstain': False, 'abstain_prob': 0.0}

        docs_for_policy = docs[: self.nq_doc_policy_max_docs]
        tail_docs = docs[self.nq_doc_policy_max_docs:]
        doc_scores = self.doc_scorer.score(query, docs_for_policy)
        score_rows = [score.to_dict() for score in doc_scores]
        features = self.nq_doc_feature_builder.build(query, docs_for_policy)
        device = next(self.nq_doc_policy.parameters()).device
        inputs = {
            'doc_features': torch.from_numpy(features).unsqueeze(0).to(device=device, dtype=torch.float32),
            'doc_mask': torch.ones(1, len(docs_for_policy), dtype=torch.bool, device=device),
        }
        with torch.no_grad():
            action = self.nq_doc_policy.select_action(
                inputs, deterministic=self.nq_doc_policy_deterministic
            )
        keep_mask = action['keep_mask'].detach().cpu().view(-1).numpy() >= 0.5
        keep_probs = action['keep_probs'].detach().cpu().view(-1).numpy()
        abstain = bool(float(action['abstain'].detach().cpu().view(-1)[0]) >= 0.5)
        abstain_prob = float(action['abstain_prob'].detach().cpu().view(-1)[0])

        kept_docs: List[Dict[str, Any]] = []
        dropped_doc_ids: List[str] = []
        for idx, doc in enumerate(docs_for_policy):
            doc_id = str(doc.get('doc_id', doc.get('id', f'doc_{idx}')))
            score_rows[idx]['nq_doc_policy_keep_prob'] = float(keep_probs[idx])
            score_rows[idx]['nq_doc_policy_kept'] = bool(keep_mask[idx] and not abstain)
            if keep_mask[idx] and not abstain:
                kept_docs.append(doc)
            else:
                dropped_doc_ids.append(doc_id)

        if abstain:
            kept_docs = []
        elif not kept_docs and docs_for_policy:
            best_idx = int(np.argmax(keep_probs))
            kept_docs = [docs_for_policy[best_idx]]
            best_id = str(docs_for_policy[best_idx].get('doc_id', docs_for_policy[best_idx].get('id', f'doc_{best_idx}')))
            dropped_doc_ids = [doc_id for doc_id in dropped_doc_ids if doc_id != best_id]
            score_rows[best_idx]['nq_doc_policy_kept'] = True

        kept_docs.extend(tail_docs)
        for idx, doc in enumerate(tail_docs, start=len(docs_for_policy)):
            score_rows.append({
                'doc_id': str(doc.get('doc_id', doc.get('id', f'doc_{idx}'))),
                'attack_prob': 0.0,
                'nq_doc_policy_keep_prob': 1.0,
                'nq_doc_policy_kept': True,
                'reasons': ['not_scored_tail_doc'],
            })
        return kept_docs, score_rows, dropped_doc_ids, {
            'abstain': abstain,
            'abstain_prob': abstain_prob,
        }

    def _rerank_docs_by_attack_score(
        self,
        docs: List[Dict[str, Any]],
        doc_scores: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Move low-risk/high-support documents first without dropping evidence."""
        if not docs or not doc_scores:
            return docs
        score_by_id = {str(row.get('doc_id')): row for row in doc_scores}

        def sort_key(item):
            doc_id = str(item.get('doc_id', item.get('id', '')))
            row = score_by_id.get(doc_id, {})
            attack_prob = float(row.get('attack_prob', 0.0))
            support = float(row.get('support_score', 0.0))
            rank = float(row.get('rank_score', 0.0))
            return (attack_prob, -support, rank)

        return sorted(docs, key=sort_key)

    def _execute_light_verify(
        self, docs: List[Dict[str, Any]], claims: List[Claim]
    ) -> List[Dict[str, Any]]:
        """执行轻量验证"""
        filtered = []
        for doc in docs:
            text = doc.get('text', doc.get('content', ''))
            # 简单检查
            if len(text) > 20:  # 过滤过短文档
                filtered.append(doc)
        return filtered

    def _execute_deep_verify(
        self,
        docs: List[Dict[str, Any]],
        claims: List[Claim],
        reports: List[ValidationReport],
    ) -> List[Dict[str, Any]]:
        """执行深度验证"""
        # 过滤掉包含冲突声明的文档
        conflict_doc_ids = set()
        for report in reports:
            if report.label == ConsistencyLabel.CONFLICT:
                for claim in report.source_claims:
                    if hasattr(claim, 'doc_id'):
                        conflict_doc_ids.add(claim.doc_id)

        filtered = []
        for doc in docs:
            doc_id = doc.get('doc_id', doc.get('id', ''))
            if doc_id not in conflict_doc_ids:
                filtered.append(doc)

        # 如果过滤太多，保留前一半
        if len(filtered) < len(docs) // 2:
            filtered = docs[:len(docs) // 2 + 1]

        return filtered

    def _execute_expand_retrieval(
        self, query: str, current_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """扩展检索"""
        # 模拟扩展检索
        # 实际实现中会调用检索器获取更多文档
        return current_docs

    def _generate_answer(
        self, query: str, docs: List[Dict[str, Any]]
    ) -> str:
        """生成答案"""
        if self.base_llm is not None:
            try:
                doc_texts = [d.get('text', d.get('content', '')) for d in docs[:5]]
                context = '\n'.join(doc_texts)
                if hasattr(self.base_llm, 'generate_answer'):
                    return self.base_llm.generate_answer(query, doc_texts)
                return self.base_llm.generate(query, context)
            except Exception:
                pass

        # 模拟答案生成
        return f"Based on the retrieved documents, the answer to '{query}' is [generated answer]."

    def _four_layer_verify(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        claims: List[Claim],
        reports: List[ValidationReport],
        answer: str,
    ) -> Dict[str, LayerResult]:
        """
        四层最终验证

        L1: Source Verification - 文档来源可信度
        L2: Evidence Verification - 引用准确性
        L3: Claim Verification - 声明一致性
        L4: Answer Verification - 答案与证据符合性
        """
        results = {}

        # L1: Source Verification
        results['source'] = self._verify_source_layer(docs)

        # L2: Evidence Verification
        results['evidence'] = self._verify_evidence_layer(docs, claims)

        # L3: Claim Verification
        results['claim'] = self._verify_claim_layer(reports)

        # L4: Answer Verification
        results['answer'] = self._verify_answer_layer(answer, claims, reports)

        return results

    def _verify_source_layer(self, docs: List[Dict[str, Any]]) -> LayerResult:
        """
        L1: Source Verification

        检查:
        - 文档来源多样性
        - 来源可信度
        - 时效性
        """
        if not docs:
            return LayerResult("L1_Source", False, {'reason': 'no_docs'}, 1.0)

        sources = [d.get('source', 'unknown') for d in docs]
        unique_sources = len(set(sources))
        diversity = unique_sources / max(len(sources), 1)

        details = {
            'num_docs': len(docs),
            'unique_sources': unique_sources,
            'source_diversity': diversity,
        }

        # 来源多样性低可能有问题
        risk_score = 0.0 if diversity > 0.5 else 0.5
        passed = diversity > 0.3

        return LayerResult("L1_Source", passed, details, risk_score)

    def _verify_evidence_layer(
        self, docs: List[Dict[str, Any]], claims: List[Claim]
    ) -> LayerResult:
        """
        L2: Evidence Verification

        检查:
        - 引用准确性（声明->原文）
        - 上下文完整性
        """
        if not claims:
            return LayerResult("L2_Evidence", True, {'reason': 'no_claims'}, 0.0)

        # 检查声明是否有来源支持
        supported = sum(1 for c in claims if c.raw_text and len(c.raw_text) > 0)
        support_ratio = supported / max(len(claims), 1)

        details = {
            'num_claims': len(claims),
            'supported_claims': supported,
            'support_ratio': support_ratio,
        }

        risk_score = 0.0 if support_ratio > 0.8 else 0.3
        passed = support_ratio > 0.5

        return LayerResult("L2_Evidence", passed, details, risk_score)

    def _verify_claim_layer(self, reports: List[ValidationReport]) -> LayerResult:
        """
        L3: Claim Verification

        检查:
        - 多源一致性
        - 冲突聚合
        """
        if not reports:
            return LayerResult("L3_Claim", True, {'reason': 'no_reports'}, 0.0)

        n_conflict = sum(1 for r in reports if r.label == ConsistencyLabel.CONFLICT)
        conflict_ratio = n_conflict / max(len(reports), 1)
        avg_risk = sum(r.risk_score for r in reports) / max(len(reports), 1)

        details = {
            'num_report_groups': len(reports),
            'conflict_groups': n_conflict,
            'conflict_ratio': conflict_ratio,
            'average_risk': avg_risk,
        }

        risk_score = avg_risk
        passed = conflict_ratio < 0.3 and avg_risk < 0.5

        return LayerResult("L3_Claim", passed, details, risk_score)

    def _verify_answer_layer(
        self, answer: str, claims: List[Claim], reports: List[ValidationReport]
    ) -> LayerResult:
        """
        L4: Answer Verification

        检查:
        - 答案与验证证据的符合性
        - 有无引入未经验证的新声明
        """
        # 简化检查: 答案非空且不与冲突声明矛盾
        if not answer or len(answer.strip()) == 0:
            return LayerResult("L4_Answer", False, {'reason': 'empty_answer'}, 1.0)

        # 检查是否有高风险冲突
        high_risk = any(r.risk_score > 0.7 for r in reports)

        details = {
            'answer_length': len(answer),
            'has_high_risk_conflict': high_risk,
        }

        risk_score = 0.5 if high_risk else 0.0
        passed = not high_risk

        return LayerResult("L4_Answer", passed, details, risk_score)

    def get_statistics(self) -> Dict[str, Any]:
        """获取防御统计"""
        if self.total_queries == 0:
            return {}

        return {
            'total_queries': self.total_queries,
            'rejected_queries': self.rejected_queries,
            'verified_queries': self.verified_queries,
            'rejection_rate': self.rejected_queries / self.total_queries,
            'verification_rate': self.verified_queries / self.total_queries,
        }
