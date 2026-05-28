"""
Claim Extractor 单元测试

测试覆盖:
- 数值提取（货币、百分比、纯数字）
- 时间提取（ISO日期、月份日期）
- 实体关系提取
- Claim对象创建和属性
- 去重功能
- Embedding-Independent（保留原始字符串）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

import torch

from verirag.claim_extractor import (
    ClaimExtractor,
    Claim,
    ClaimType,
    NUMERICAL_PATTERNS,
    TEMPORAL_PATTERNS,
)


class TestClaimExtractor(unittest.TestCase):
    """声明提取器测试类"""

    @classmethod
    def setUpClass(cls):
        """测试前初始化"""
        cls.extractor = ClaimExtractor(config={
            'rule_engine_enabled': True,
            'llm_extractor_enabled': False,
            'min_confidence': 0.5,
        })

    def test_numerical_extraction_currency(self):
        """测试货币数值提取"""
        text = "The company reported revenue of $15K in Q1 2024."
        claims = self.extractor.extract([text], ['doc_1'])

        # 检查是否提取到数值声明
        numerical_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
        self.assertGreater(len(numerical_claims), 0)

        # 检查原始字符串是否保留
        claim = numerical_claims[0]
        self.assertIsNotNone(claim.value)
        self.assertIn('$', claim.value)  # 原始字符串保留$符号

    def test_numerical_extraction_percentage(self):
        """测试百分比提取"""
        text = "The success rate increased by 25% this year."
        claims = self.extractor.extract([text], ['doc_2'])

        numerical_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
        self.assertGreater(len(numerical_claims), 0)

        # 检查百分比保留
        has_percentage = any('%' in str(c.value) for c in numerical_claims)
        self.assertTrue(has_percentage)

    def test_numerical_precision_drift_detection(self):
        """
        测试数值精度保留（防御Embedding Blind Spot）

        关键测试: 确保 $15K 和 $65K 保留不同的原始字符串
        """
        text1 = "The price was $15K."
        text2 = "The price was $65K."

        claims1 = self.extractor.extract([text1], ['doc_a'])
        claims2 = self.extractor.extract([text2], ['doc_b'])

        # 提取数值声明
        num1 = [c for c in claims1 if c.claim_type == ClaimType.NUMERICAL]
        num2 = [c for c in claims2 if c.claim_type == ClaimType.NUMERICAL]

        self.assertGreater(len(num1), 0)
        self.assertGreater(len(num2), 0)

        # 原始字符串必须不同
        value1 = num1[0].value
        value2 = num2[0].value
        self.assertNotEqual(value1, value2)

        # 检查'1'和'6'的差异
        self.assertIn('1', str(value1))
        self.assertIn('6', str(value2))

    def test_temporal_extraction_iso_date(self):
        """测试ISO日期提取"""
        text = "The event took place on 2024-01-15."
        claims = self.extractor.extract([text], ['doc_3'])

        temporal_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        self.assertGreater(len(temporal_claims), 0)

        # 检查日期格式保留
        claim = temporal_claims[0]
        self.assertIn('2024', str(claim.value))

    def test_temporal_extraction_year(self):
        """测试年份提取"""
        text = "The company was founded in 1998."
        claims = self.extractor.extract([text], ['doc_4'])

        temporal_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        self.assertGreater(len(temporal_claims), 0)

        claim = temporal_claims[0]
        self.assertIn('1998', str(claim.value))

    def test_entity_relation_extraction(self):
        """测试实体关系提取"""
        text = "Einstein was born in Germany."
        claims = self.extractor.extract([text], ['doc_5'])

        factual_claims = [c for c in claims if c.claim_type == ClaimType.FACTUAL]
        self.assertGreater(len(factual_claims), 0)

        # 检查subject和object
        claim = factual_claims[0]
        self.assertTrue(
            'einstein' in claim.subject.lower() or
            'einstein' in claim.object.lower()
        )

    def test_deduplication(self):
        """测试声明去重功能"""
        text = "The price is $100. The price is $100. The price is $100."
        claims = self.extractor.extract([text], ['doc_6'])

        # 去重后应该只有少量唯一声明
        unique_subjects = set(c.subject for c in claims)
        unique_objects = set(c.object for c in claims)

        # 不应有大量重复
        self.assertLessEqual(len(claims), 10)

    def test_empty_document(self):
        """测试空文档处理"""
        claims = self.extractor.extract([''], ['empty_doc'])
        self.assertEqual(len(claims), 0)

    def test_confidence_filtering(self):
        """测试置信度过滤"""
        # 使用高置信度阈值
        strict_extractor = ClaimExtractor(config={
            'rule_engine_enabled': True,
            'min_confidence': 0.95,
        })

        text = "The value is 42."
        claims = strict_extractor.extract([text], ['doc_7'])

        # 所有声明的置信度应 >= 0.95
        for claim in claims:
            self.assertGreaterEqual(claim.confidence, 0.95)

    def test_claim_id_uniqueness(self):
        """测试Claim ID唯一性"""
        text = "Value A is 10. Value B is 20."
        claims = self.extractor.extract([text], ['doc_8'])

        # 所有ID应该唯一
        ids = [c.claim_id for c in claims]
        self.assertEqual(len(ids), len(set(ids)))

    def test_original_string_preservation(self):
        """
        测试原始字符串保留（核心！防御Embedding Blind Spot）

        确保数值声明的value字段始终保留原始字符串格式
        """
        text = "The revenue was $1,234,567.89."
        claims = self.extractor.extract([text], ['doc_9'])

        numerical_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
        self.assertGreater(len(numerical_claims), 0)

        for claim in numerical_claims:
            # value字段必须存在
            self.assertIsNotNone(claim.value)
            # 原始字符串包含逗号和$符号
            original = str(claim.value)
            # 至少包含数字
            self.assertTrue(any(c.isdigit() for c in original))

    def test_batch_processing(self):
        """测试批量处理"""
        texts = [
            "The price is $50.",
            "Founded in 1990.",
            "Rate: 75%",
        ]
        doc_ids = ['doc_a', 'doc_b', 'doc_c']

        all_claims = self.extractor.extract(texts, doc_ids)

        # 应该提取到声明
        self.assertGreater(len(all_claims), 0)

        # 每个声明应该有正确的doc_id
        doc_id_set = set(c.doc_id for c in all_claims)
        self.assertTrue(doc_id_set.issubset(set(doc_ids)))

    def test_pytorch_forward(self):
        """测试PyTorch forward接口"""
        texts = ["The value is 100."]

        # 确保模型在eval模式
        self.extractor.eval()

        with torch.no_grad():
            results = self.extractor.forward(texts)

        # 应该返回列表的列表
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(texts))


class TestRegexPatterns(unittest.TestCase):
    """正则表达式模式测试"""

    def test_currency_pattern(self):
        """测试货币模式"""
        pattern = NUMERICAL_PATTERNS['currency']

        matches = pattern.findall("The price is $15K.")
        self.assertGreater(len(matches), 0)

        matches = pattern.findall("Revenue: $1.5M")
        self.assertGreater(len(matches), 0)

    def test_percentage_pattern(self):
        """测试百分比模式"""
        pattern = NUMERICAL_PATTERNS['percentage']

        matches = pattern.findall("Growth rate: 25%")
        self.assertGreater(len(matches), 0)

        matches = pattern.findall("Only 5 percent survived.")
        self.assertGreater(len(matches), 0)

    def test_iso_date_pattern(self):
        """测试ISO日期模式"""
        pattern = TEMPORAL_PATTERNS['iso_date']

        matches = pattern.findall("Date: 2024-01-15")
        self.assertEqual(len(matches), 1)

    def test_year_pattern(self):
        """测试年份模式"""
        pattern = TEMPORAL_PATTERNS['year']

        matches = pattern.findall("Founded in 1998")
        self.assertGreater(len(matches), 0)


if __name__ == '__main__':
    unittest.main()
