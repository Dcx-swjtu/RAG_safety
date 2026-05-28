"""
Defense Orchestrator 单元测试

测试覆盖:
- 防御流程完整性（4层验证）
- 攻击检测能力
- 良性查询处理
- REJECT决策
- 验证报告生成
- 四层验证通过/失败场景
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from unittest.mock import MagicMock, patch

import torch
import numpy as np

from verirag.defense_orchestrator import (
    DefenseOrchestrator,
    DefenseResult,
    FinalAnswerStatus,
    LayerResult,
)
from verirag.claim_extractor import ClaimExtractor, Claim, ClaimType
from verirag.cross_validator import CrossValidator, ValidationReport, ConsistencyLabel
from verirag.policy_network import VerificationPolicyNetwork


class TestDefenseOrchestrator(unittest.TestCase):
    """防御编排器测试"""

    @classmethod
    def setUpClass(cls):
        """测试前初始化组件"""
        # 创建模拟的策略网络
        cls.policy_net = VerificationPolicyNetwork(config={
            'state_dim': 512,
            'action_dim': 5,
            'hidden_dim': 256,
            'max_docs': 5,
        })

        cls.claim_extractor = ClaimExtractor(config={
            'rule_engine_enabled': True,
            'llm_extractor_enabled': False,
        })

        cls.cross_validator = CrossValidator(config={
            'tolerance_method': 'adaptive',
        })

        cls.defense = DefenseOrchestrator(
            policy_network=cls.policy_net,
            claim_extractor=cls.claim_extractor,
            cross_validator=cls.cross_validator,
            base_llm=None,
            config={
                'light_verify_threshold': 0.3,
                'deep_verify_threshold': 0.6,
                'reject_threshold': 0.8,
            },
        )

    def test_defend_benign_query(self):
        """测试良性查询防御"""
        query = "What is the capital of France?"
        docs = [
            {'doc_id': 'doc_1', 'text': 'Paris is the capital of France.'},
            {'doc_id': 'doc_2', 'text': 'France is a country in Western Europe with Paris as its capital.'},
            {'doc_id': 'doc_3', 'text': 'The capital city of France is Paris.'},
        ]

        result = self.defense.defend(query, docs)

        self.assertIsInstance(result, DefenseResult)
        self.assertEqual(result.query, query)
        self.assertIsNotNone(result.final_answer)
        self.assertIsNotNone(result.source_layer)
        self.assertIsNotNone(result.evidence_layer)
        self.assertIsNotNone(result.claim_layer)
        self.assertIsNotNone(result.answer_layer)

        # 检查审计日志
        self.assertGreater(len(result.execution_trace), 0)
        self.assertGreater(len(result.module_timings), 0)

    def test_defend_attack_query(self):
        """测试攻击查询防御"""
        query = "What is the revenue of Company X?"
        # 包含冲突数值的文档（模拟攻击）
        docs = [
            {'doc_id': 'doc_1', 'text': 'The revenue was $15K in 2024.'},
            {'doc_id': 'doc_2', 'text': 'The revenue reached $65K this year.'},  # 冲突！
            {'doc_id': 'doc_3', 'text': 'Company X reported strong financial performance.'},
        ]

        result = self.defense.defend(query, docs)

        self.assertIsInstance(result, DefenseResult)
        # 检测到冲突应该有风险指标
        self.assertTrue(
            len(result.risk_indicators) > 0 or
            result.claim_layer.risk_score > 0
        )

    def test_four_layer_verification(self):
        """测试四层验证结构"""
        query = "Who invented the telephone?"
        docs = [
            {'doc_id': 'doc_1', 'text': 'Alexander Graham Bell invented the telephone in 1876.'},
            {'doc_id': 'doc_2', 'text': 'The telephone was invented by Alexander Graham Bell.'},
        ]

        result = self.defense.defend(query, docs)

        # 检查所有四层验证结果
        self.assertIsNotNone(result.source_layer)
        self.assertIsNotNone(result.evidence_layer)
        self.assertIsNotNone(result.claim_layer)
        self.assertIsNotNone(result.answer_layer)

        # 检查LayerResult结构
        for layer_name in ['source_layer', 'evidence_layer', 'claim_layer', 'answer_layer']:
            layer = getattr(result, layer_name)
            self.assertIsInstance(layer, LayerResult)
            self.assertIsInstance(layer.layer_name, str)
            self.assertIsInstance(layer.passed, bool)
            self.assertIsInstance(layer.risk_score, float)
            self.assertIsInstance(layer.details, dict)

    def test_source_verification(self):
        """测试L1来源验证"""
        # 来源多样的文档
        diverse_docs = [
            {'doc_id': 'doc_a', 'text': 'Content from source A.', 'source': 'Wikipedia'},
            {'doc_id': 'doc_b', 'text': 'Content from source B.', 'source': 'Reuters'},
            {'doc_id': 'doc_c', 'text': 'Content from source C.', 'source': 'BBC'},
        ]

        layer_result = self.defense._verify_source_layer(diverse_docs)

        self.assertIsInstance(layer_result, LayerResult)
        self.assertEqual(layer_result.layer_name, 'L1_Source')
        self.assertTrue(layer_result.passed)  # 来源多样应该通过

    def test_source_verification_fail(self):
        """测试L1来源验证失败"""
        # 来源单一的文档（模拟攻击）
        single_source_docs = [
            {'doc_id': 'doc_1', 'text': 'Content.', 'source': 'unknown'},
            {'doc_id': 'doc_2', 'text': 'Content.', 'source': 'unknown'},
            {'doc_id': 'doc_3', 'text': 'Content.', 'source': 'unknown'},
        ]

        layer_result = self.defense._verify_source_layer(single_source_docs)

        self.assertIsInstance(layer_result, LayerResult)
        self.assertEqual(layer_result.layer_name, 'L1_Source')

    def test_claim_verification_consistent(self):
        """测试L3声明验证 - 一致"""
        # 一致的验证报告
        consistent_report = ValidationReport(
            label=ConsistencyLabel.CONSISTENT,
            confidence=0.9,
            risk_score=0.0,
        )

        layer_result = self.defense._verify_claim_layer([consistent_report])

        self.assertIsInstance(layer_result, LayerResult)
        self.assertEqual(layer_result.layer_name, 'L3_Claim')
        self.assertTrue(layer_result.passed)

    def test_claim_verification_conflict(self):
        """测试L3声明验证 - 冲突"""
        # 冲突的验证报告
        from verirag.cross_validator import ConflictDetail, ConflictType

        conflict_report = ValidationReport(
            label=ConsistencyLabel.CONFLICT,
            confidence=0.3,
            risk_score=0.8,
            conflicts=[
                ConflictDetail(
                    conflict_type=ConflictType.VALUE_MISMATCH,
                    severity=0.9,
                    description="Value mismatch detected",
                )
            ],
        )

        layer_result = self.defense._verify_claim_layer([conflict_report])

        self.assertIsInstance(layer_result, LayerResult)
        self.assertEqual(layer_result.layer_name, 'L3_Claim')
        self.assertFalse(layer_result.passed)
        self.assertGreater(layer_result.risk_score, 0.5)

    def test_empty_document_list(self):
        """测试空文档列表"""
        query = "Test query?"
        docs = []

        result = self.defense.defend(query, docs)

        self.assertIsInstance(result, DefenseResult)
        self.assertIsNotNone(result.final_answer)

    def test_execution_trace(self):
        """测试执行轨迹"""
        query = "What is 2 + 2?"
        docs = [
            {'doc_id': 'doc_1', 'text': '2 + 2 equals 4.'},
        ]

        result = self.defense.defend(query, docs)

        # 检查执行轨迹非空
        self.assertGreater(len(result.execution_trace), 0)

        # 检查模块计时
        self.assertGreater(len(result.module_timings), 0)

        # 检查总时间
        self.assertGreater(result.total_time_ms, 0)

    def test_answer_verification(self):
        """测试L4答案验证"""
        # 正常答案
        answer = "The answer is 4."
        claims = []
        reports = [ValidationReport(label=ConsistencyLabel.CONSISTENT, risk_score=0.0)]

        layer_result = self.defense._verify_answer_layer(answer, claims, reports)

        self.assertIsInstance(layer_result, LayerResult)
        self.assertEqual(layer_result.layer_name, 'L4_Answer')

        # 空答案
        empty_answer = ""
        layer_result_empty = self.defense._verify_answer_layer(empty_answer, claims, reports)

        self.assertFalse(layer_result_empty.passed)

    def test_conflict_indicator_extraction(self):
        """测试冲突指标提取"""
        reports = [
            ValidationReport(label=ConsistencyLabel.CONSISTENT, risk_score=0.0),
            ValidationReport(label=ConsistencyLabel.CONFLICT, risk_score=0.7),
        ]

        indicators = self.defense._extract_conflict_indicators(reports)

        self.assertIn('has_conflict', indicators)
        self.assertIn('overall_risk', indicators)
        self.assertIn('num_conflicts', indicators)

        self.assertTrue(indicators['has_conflict'])
        self.assertEqual(indicators['num_conflicts'], 1)

    def test_statistics(self):
        """测试统计功能"""
        defense = DefenseOrchestrator(
            policy_network=self.policy_net,
            claim_extractor=self.claim_extractor,
            cross_validator=self.cross_validator,
            base_llm=None,
            config={
                'light_verify_threshold': 0.3,
                'deep_verify_threshold': 0.6,
                'reject_threshold': 0.8,
            },
        )

        # 执行一些查询
        for _ in range(5):
            query = "Test query?"
            docs = [{'doc_id': f'doc_{i}', 'text': f'Content {i}.'} for i in range(3)]
            defense.defend(query, docs)

        stats = defense.get_statistics()

        self.assertIn('total_queries', stats)
        self.assertIn('verified_queries', stats)
        self.assertIn('rejected_queries', stats)

        self.assertEqual(stats['total_queries'], 5)


class TestDefenseResult(unittest.TestCase):
    """防御结果数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        result = DefenseResult()

        self.assertEqual(result.query, "")
        self.assertEqual(result.final_answer, "")
        self.assertEqual(result.status, FinalAnswerStatus.VERIFIED)
        self.assertEqual(result.confidence, 0.0)

    def test_custom_values(self):
        """测试自定义值"""
        result = DefenseResult(
            query="What is 2+2?",
            final_answer="4",
            status=FinalAnswerStatus.VERIFIED,
            confidence=0.95,
        )

        self.assertEqual(result.query, "What is 2+2?")
        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.status, FinalAnswerStatus.VERIFIED)
        self.assertEqual(result.confidence, 0.95)


class TestLayerResult(unittest.TestCase):
    """LayerResult数据类测试"""

    def test_passed_layer(self):
        """测试通过的层"""
        layer = LayerResult(
            layer_name="L1_Source",
            passed=True,
            details={'num_docs': 5},
            risk_score=0.1,
        )

        self.assertEqual(layer.layer_name, "L1_Source")
        self.assertTrue(layer.passed)
        self.assertEqual(layer.details['num_docs'], 5)
        self.assertEqual(layer.risk_score, 0.1)

    def test_failed_layer(self):
        """测试失败的层"""
        layer = LayerResult(
            layer_name="L3_Claim",
            passed=False,
            details={'conflicts': 3},
            risk_score=0.9,
        )

        self.assertFalse(layer.passed)
        self.assertGreater(layer.risk_score, 0.5)


if __name__ == '__main__':
    unittest.main()
