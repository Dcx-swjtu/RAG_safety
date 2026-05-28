"""
Module 2: Cross-Validator（多源交叉验证器）

核心创新: Embedding-Independent验证
- 数值比较: 直接字符串比对（解决Embedding Blind Spot）
- 实体比较: 同义词/别名处理
- 时序比较: 时间线一致性
- 因果比较: 逻辑蕴含关系

验证流程:
1. 按(entity, attribute)分组声明
2. 组内一致性检查（符号级比对）
3. 冲突聚合与风险评估
4. 生成ValidationReport
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto

import torch
import torch.nn as nn
import numpy as np


# ==================== 数据类型定义 ====================

class ConsistencyLabel(Enum):
    """一致性标签"""
    CONSISTENT = "consistent"       # 多源完全一致
    APPROXIMATE = "approximate"     # 近似一致（容差内）
    CONFLICT = "conflict"           # 多源冲突（攻击信号）
    UNCERTAIN = "uncertain"         # 信息不足
    SINGLE_SOURCE = "single_source" # 仅单源信息


class ConflictType(Enum):
    """冲突类型枚举"""
    VALUE_MISMATCH = "value_mismatch"           # 数值不一致
    SIGN_FLIP = "sign_flip"                     # 符号翻转
    ENTITY_SUBSTITUTION = "entity_substitution" # 实体替换
    TEMPORAL_CONFLICT = "temporal_conflict"     # 时间冲突
    RANGE_VIOLATION = "range_violation"         # 范围违规
    SEMANTIC_INVERSION = "semantic_inversion"   # 语义反转
    PRECISION_DRIFT = "precision_drift"         # 精度漂移（Embedding Blind Spot）
    SOURCE_DISAGREEMENT = "source_disagreement" # 来源不一致


@dataclass
class ConflictDetail:
    """冲突详情"""
    conflict_type: ConflictType
    severity: float                              # 严重程度 [0, 1]
    source_claims: List[str] = field(default_factory=list)  # 冲突的claim_id列表
    description: str = ""


@dataclass
class ValidationReport:
    """验证报告"""
    label: ConsistencyLabel = ConsistencyLabel.SINGLE_SOURCE
    confidence: float = 0.0
    risk_score: float = 0.0                      # 0-1, 越高越可疑
    conflicts: List[ConflictDetail] = field(default_factory=list)
    exact_match: bool = False
    tolerance_check_passed: bool = False
    precision_drift_detected: bool = False
    source_claims: List[Any] = field(default_factory=list)


# ==================== 同义词/别名映射 ====================

# 常用实体别名映射（可扩展）
ENTITY_ALIASES = {
    'us': ['usa', 'united states', 'america', 'united states of america'],
    'usa': ['us', 'united states', 'america'],
    'uk': ['united kingdom', 'britain', 'great britain'],
    'eu': ['european union'],
    'un': ['united nations'],
    'nasa': ['national aeronautics and space administration'],
    'fbi': ['federal bureau of investigation'],
    'cia': ['central intelligence agency'],
}

# 语义反转关键词对
ANTONYM_PAIRS = [
    ('increase', 'decrease'),
    ('rise', 'fall'),
    ('grow', 'shrink'),
    ('higher', 'lower'),
    ('more', 'less'),
    ('larger', 'smaller'),
    ('earlier', 'later'),
    ('before', 'after'),
    ('superior', 'inferior'),
    ('positive', 'negative'),
    ('success', 'failure'),
    ('gain', 'loss'),
    ('accept', 'reject'),
    ('approve', 'deny'),
    ('true', 'false'),
]


# ==================== CrossValidator 主类 ====================

class CrossValidator:
    """
    多源交叉验证器: 跨文档验证声明一致性

    核心机制:
    1. Embedding-Independent: 使用符号级比对，不依赖embedding相似度
    2. 数值精度感知: 检测$15K -> $65K类型攻击
    3. 语义等价性: 基于符号逻辑判断（非embedding!）
    4. 冲突聚合: 综合计算攻击风险评分
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化交叉验证器

        Args:
            config: 配置字典
                - tolerance_method: str ("absolute"/"relative"/"adaptive")
                - default_tolerance: float (默认0.01)
                - semantic_check_enabled: bool (默认True)
                - numerical_strict_mode: bool (默认True)
                - precision_drift_threshold: float (默认0.1)
        """
        self.config = config or {}
        self.tolerance_method = self.config.get('tolerance_method', 'adaptive')
        self.default_tolerance = self.config.get('default_tolerance', 0.01)
        self.semantic_check_enabled = self.config.get('semantic_check_enabled', True)
        self.numerical_strict_mode = self.config.get('numerical_strict_mode', True)
        self.precision_drift_threshold = self.config.get('precision_drift_threshold', 0.1)

    def validate(self, claims: List[Any]) -> ValidationReport:
        """
        对声明列表进行交叉验证

        Args:
            claims: Claim对象列表（来自多个文档）

        Returns:
            ValidationReport包含一致性标签和风险评分
        """
        if len(claims) == 0:
            return ValidationReport(
                label=ConsistencyLabel.UNCERTAIN,
                confidence=0.0,
                risk_score=0.0,
            )

        if len(claims) == 1:
            return ValidationReport(
                label=ConsistencyLabel.SINGLE_SOURCE,
                confidence=claims[0].confidence if hasattr(claims[0], 'confidence') else 0.5,
                risk_score=0.2,  # 单源有一定风险
                source_claims=claims,
            )

        # 按声明类型分组验证
        claim_type = claims[0].claim_type if hasattr(claims[0], 'claim_type') else None

        if claim_type is not None and claim_type.value == 'numerical':
            return self._validate_numerical_claims(claims)
        elif claim_type is not None and claim_type.value == 'temporal':
            return self._validate_temporal_claims(claims)
        elif claim_type is not None and claim_type.value == 'causal':
            return self._validate_causal_claims(claims)
        else:
            return self._validate_factual_claims(claims)

    def validate_batch(self, claim_groups: List[List[Any]]) -> List[ValidationReport]:
        """
        批量验证多个声明组

        Args:
            claim_groups: 声明组列表，每组包含同一声明的多个来源

        Returns:
            ValidationReport列表
        """
        return [self.validate(group) for group in claim_groups]

    def _validate_numerical_claims(self, claims: List[Any]) -> ValidationReport:
        """
        数值声明验证（核心！防御Embedding Blind Spot）

        验证流程:
        1. 精确匹配检查（原始字符串比对）
        2. Embedding Blind Spot检测（emb_sim高但val_diff大）
        3. 容差检查（自适应容差）
        4. 冲突聚合
        """
        values = []
        raw_strings = []
        for c in claims:
            if hasattr(c, 'value') and c.value is not None:
                raw_strings.append(str(c.value))
                # 提取数值
                num = self._extract_number(str(c.value))
                if num is not None:
                    values.append((num, c))

        if len(values) < 2:
            return ValidationReport(
                label=ConsistencyLabel.SINGLE_SOURCE,
                confidence=0.5,
                risk_score=0.2,
                source_claims=claims,
            )

        # 1. 精确匹配检查（原始字符串）
        if len(set(raw_strings)) == 1:
            return ValidationReport(
                label=ConsistencyLabel.CONSISTENT,
                confidence=0.95,
                risk_score=0.0,
                exact_match=True,
                tolerance_check_passed=True,
                source_claims=claims,
            )

        # 2. & 3. 逐对比较
        conflicts = []
        all_match = True
        precision_drift_detected = False

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                val_i, claim_i = values[i]
                val_j, claim_j = values[j]

                # 计算数值差异（相对差异）
                val_diff = abs(val_i - val_j) / max(abs(val_i), abs(val_j), 1e-8)

                # Embedding Blind Spot检测:
                # 如果原始文本embedding相似度高但数值差异大
                raw_i = str(claim_i.value) if hasattr(claim_i, 'value') else ""
                raw_j = str(claim_j.value) if hasattr(claim_j, 'value') else ""

                # 文本相似度（字符级）
                text_sim = self._char_level_similarity(raw_i, raw_j)

                # 如果文本很相似但数值差异大 -> PRECISION_DRIFT攻击
                if text_sim > 0.8 and val_diff > self.precision_drift_threshold:
                    conflicts.append(ConflictDetail(
                        conflict_type=ConflictType.PRECISION_DRIFT,
                        severity=min(val_diff, 1.0),
                        source_claims=[
                            claim_i.claim_id if hasattr(claim_i, 'claim_id') else str(i),
                            claim_j.claim_id if hasattr(claim_j, 'claim_id') else str(j),
                        ],
                        description=f"Embedding Blind Spot detected: '{raw_i}' vs '{raw_j}', "
                                   f"text_sim={text_sim:.3f}, val_diff={val_diff:.3f}",
                    ))
                    precision_drift_detected = True
                    all_match = False
                    continue

                # 容差检查
                tolerance = self._compute_tolerance(val_i, val_j)
                if val_diff <= tolerance:
                    continue  # 在容差内，一致
                else:
                    all_match = False
                    conflicts.append(ConflictDetail(
                        conflict_type=ConflictType.VALUE_MISMATCH,
                        severity=min(val_diff, 1.0),
                        source_claims=[
                            claim_i.claim_id if hasattr(claim_i, 'claim_id') else str(i),
                            claim_j.claim_id if hasattr(claim_j, 'claim_id') else str(j),
                        ],
                        description=f"Value mismatch: {val_i} vs {val_j} (diff={val_diff:.3f})",
                    ))

        # 4. 综合判断
        risk_score = self._compute_risk_score(conflicts, len(claims))

        if precision_drift_detected:
            label = ConsistencyLabel.CONFLICT
        elif all_match:
            label = ConsistencyLabel.CONSISTENT
        elif any(c.severity > 0.5 for c in conflicts):
            label = ConsistencyLabel.CONFLICT
        elif conflicts:
            label = ConsistencyLabel.APPROXIMATE
        else:
            label = ConsistencyLabel.CONSISTENT

        return ValidationReport(
            label=label,
            confidence=1.0 - risk_score,
            risk_score=risk_score,
            conflicts=conflicts,
            exact_match=all_match and len(conflicts) == 0,
            tolerance_check_passed=not any(c.conflict_type == ConflictType.VALUE_MISMATCH
                                          for c in conflicts),
            precision_drift_detected=precision_drift_detected,
            source_claims=claims,
        )

    def _validate_temporal_claims(self, claims: List[Any]) -> ValidationReport:
        """时间声明验证: 检查时间线一致性"""
        # 简化实现: 比较时间字符串
        time_values = []
        for c in claims:
            if hasattr(c, 'value') and c.value:
                time_values.append(str(c.value))

        if len(time_values) < 2:
            return ValidationReport(
                label=ConsistencyLabel.SINGLE_SOURCE,
                confidence=0.5,
                risk_score=0.2,
                source_claims=claims,
            )

        # 检查是否完全一致
        if len(set(time_values)) == 1:
            return ValidationReport(
                label=ConsistencyLabel.CONSISTENT,
                confidence=0.9,
                risk_score=0.0,
                exact_match=True,
                source_claims=claims,
            )

        # 检查时间冲突
        conflicts = []
        # 提取年份进行比对
        years_list = []
        for tv in time_values:
            year_match = re.search(r'\b(\d{4})\b', tv)
            if year_match:
                years_list.append(int(year_match.group(1)))

        if len(years_list) >= 2:
            for i in range(len(years_list)):
                for j in range(i + 1, len(years_list)):
                    if abs(years_list[i] - years_list[j]) > 1:  # 超过1年的差异
                        conflicts.append(ConflictDetail(
                            conflict_type=ConflictType.TEMPORAL_CONFLICT,
                            severity=min(abs(years_list[i] - years_list[j]) / 100.0, 1.0),
                            description=f"Temporal conflict: {years_list[i]} vs {years_list[j]}",
                        ))

        risk_score = self._compute_risk_score(conflicts, len(claims))
        label = ConsistencyLabel.CONFLICT if conflicts else ConsistencyLabel.APPROXIMATE

        return ValidationReport(
            label=label,
            confidence=1.0 - risk_score,
            risk_score=risk_score,
            conflicts=conflicts,
            source_claims=claims,
        )

    def _validate_causal_claims(self, claims: List[Any]) -> ValidationReport:
        """因果声明验证: 检查逻辑蕴含关系"""
        # 因果声明较为复杂，主要检查是否存在语义反转
        conflicts = []

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                raw_i = str(claims[i].raw_text) if hasattr(claims[i], 'raw_text') else ""
                raw_j = str(claims[j].raw_text) if hasattr(claims[j], 'raw_text') else ""

                # 检查反义词对（语义反转）
                if self._check_antonym_presence(raw_i, raw_j):
                    conflicts.append(ConflictDetail(
                        conflict_type=ConflictType.SEMANTIC_INVERSION,
                        severity=0.8,
                        description=f"Semantic inversion detected between causal claims",
                    ))

        risk_score = self._compute_risk_score(conflicts, len(claims))
        label = ConsistencyLabel.CONFLICT if conflicts else ConsistencyLabel.CONSISTENT

        return ValidationReport(
            label=label,
            confidence=1.0 - risk_score,
            risk_score=risk_score,
            conflicts=conflicts,
            source_claims=claims,
        )

    def _validate_factual_claims(self, claims: List[Any]) -> ValidationReport:
        """
        事实声明验证: 主体-谓词-客体匹配

        检查:
        - 实体是否一致（考虑别名）
        - 是否存在实体替换攻击
        """
        subjects = []
        objects = []
        for c in claims:
            if hasattr(c, 'subject'):
                subjects.append(c.subject.lower().strip())
            if hasattr(c, 'object'):
                objects.append(c.object.lower().strip())

        conflicts = []

        # 检查实体一致性
        if len(subjects) >= 2:
            for i in range(len(subjects)):
                for j in range(i + 1, len(subjects)):
                    if not self._entities_match(subjects[i], subjects[j]):
                        conflicts.append(ConflictDetail(
                            conflict_type=ConflictType.ENTITY_SUBSTITUTION,
                            severity=0.7,
                            description=f"Entity substitution: '{subjects[i]}' vs '{subjects[j]}'",
                        ))

        if len(objects) >= 2:
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    if not self._entities_match(objects[i], objects[j]):
                        conflicts.append(ConflictDetail(
                            conflict_type=ConflictType.ENTITY_SUBSTITUTION,
                            severity=0.7,
                            description=f"Object substitution: '{objects[i]}' vs '{objects[j]}'",
                        ))

        # 检查语义反转
        raw_texts = [str(c.raw_text) for c in claims if hasattr(c, 'raw_text')]
        for i in range(len(raw_texts)):
            for j in range(i + 1, len(raw_texts)):
                if self._check_antonym_presence(raw_texts[i], raw_texts[j]):
                    conflicts.append(ConflictDetail(
                        conflict_type=ConflictType.SEMANTIC_INVERSION,
                        severity=0.8,
                        description="Semantic inversion in factual claims",
                    ))

        risk_score = self._compute_risk_score(conflicts, len(claims))

        if not conflicts:
            return ValidationReport(
                label=ConsistencyLabel.CONSISTENT,
                confidence=0.85,
                risk_score=0.0,
                exact_match=True,
                source_claims=claims,
            )

        label = ConsistencyLabel.CONFLICT if any(c.severity > 0.5 for c in conflicts) else ConsistencyLabel.APPROXIMATE

        return ValidationReport(
            label=label,
            confidence=1.0 - risk_score,
            risk_score=risk_score,
            conflicts=conflicts,
            source_claims=claims,
        )

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """从文本中提取数值"""
        # 处理货币格式 $15K -> 15000
        text = text.strip()

        # 处理K/M/B/T后缀
        multiplier = 1.0
        if text.endswith('K') or text.endswith('k'):
            multiplier = 1e3
            text = text[:-1]
        elif text.endswith('M') or text.endswith('m'):
            multiplier = 1e6
            text = text[:-1]
        elif text.endswith('B') or text.endswith('b'):
            multiplier = 1e9
            text = text[:-1]
        elif text.endswith('T') or text.endswith('t'):
            multiplier = 1e12
            text = text[:-1]

        # 提取数字
        number_match = re.search(r'[\$\€\£\¥]?\s*([\d,]+(?:\.\d+)?)', text)
        if number_match:
            try:
                return float(number_match.group(1).replace(',', '')) * multiplier
            except ValueError:
                return None
        return None

    def _compute_tolerance(self, val1: float, val2: float) -> float:
        """
        计算自适应容差

        策略:
        - absolute: 固定容差
        - relative: 相对容差（基于数值量级）
        - adaptive: 自适应容差（大额更宽松，小额更严格）
        """
        if self.tolerance_method == 'absolute':
            return self.default_tolerance

        avg_val = (abs(val1) + abs(val2)) / 2.0

        if self.tolerance_method == 'relative':
            return self.default_tolerance * max(avg_val, 1.0)

        # adaptive: 基于数值量级的自适应容差
        if avg_val < 1.0:
            return 0.001  # 小于1: 千分之一精度
        elif avg_val < 100.0:
            return 0.01   # 小于100: 百分之一精度
        elif avg_val < 10000.0:
            return 0.05   # 小于10000: 百分之五精度
        else:
            return 0.1    # 大于10000: 十分之一精度

    @staticmethod
    def _char_level_similarity(str1: str, str2: str) -> float:
        """
        字符级相似度（用于Embedding Blind Spot检测）

        计算两个字符串的字符重叠度
        """
        if not str1 or not str2:
            return 0.0

        # 归一化: 移除空格和标点，转小写
        s1 = re.sub(r'[^\w]', '', str1.lower())
        s2 = re.sub(r'[^\w]', '', str2.lower())

        if not s1 or not s2:
            return 0.0

        # 最长公共子序列近似（使用字符集合Jaccard）
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _entities_match(self, entity1: str, entity2: str) -> bool:
        """
        检查两个实体是否匹配（考虑别名）

        支持:
        - 精确匹配
        - 别名映射
        - 子串匹配（简化版）
        """
        e1 = entity1.lower().strip()
        e2 = entity2.lower().strip()

        # 精确匹配
        if e1 == e2:
            return True

        # 检查别名
        for canonical, aliases in ENTITY_ALIASES.items():
            alias_set = set([canonical] + [a.lower() for a in aliases])
            if e1 in alias_set and e2 in alias_set:
                return True

        # 子串匹配（一个包含另一个）
        if e1 in e2 or e2 in e1:
            return True

        return False

    @staticmethod
    def _check_antonym_presence(text1: str, text2: str) -> bool:
        """
        检查两段文本中是否包含反义词对（语义反转检测）

        返回True如果发现任何反义词对
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        for word1, word2 in ANTONYM_PAIRS:
            # 检查word1在text1中且word2在text2中（或反之）
            if (word1 in text1_lower and word2 in text2_lower) or \
               (word2 in text1_lower and word1 in text2_lower):
                return True

        return False

    @staticmethod
    def _compute_risk_score(conflicts: List[ConflictDetail], num_sources: int) -> float:
        """
        计算综合风险评分

        考虑因素:
        - 冲突数量和严重程度
        - 来源数量
        - 冲突类型（PRECISION_DRIFT权重最高）
        """
        if not conflicts:
            return 0.0

        base_score = sum(c.severity for c in conflicts) / max(len(conflicts), 1)

        # PRECISION_DRIFT权重加倍
        if any(c.conflict_type == ConflictType.PRECISION_DRIFT for c in conflicts):
            base_score *= 1.5

        # 来源越多但仍有冲突，风险越高
        source_factor = min(num_sources / 5.0, 1.0)

        return min(base_score * source_factor, 1.0)

    def group_claims_by_key(self, claims: List[Any]) -> Dict[str, List[Any]]:
        """
        按(subject, predicate, claim_type)分组声明

        这是交叉验证的前置步骤
        """
        groups = {}
        for claim in claims:
            if not hasattr(claim, 'subject') or not hasattr(claim, 'predicate'):
                continue
            subject = claim.subject.lower().strip()
            predicate = claim.predicate.lower().strip()
            c_type = claim.claim_type.value if hasattr(claim, 'claim_type') else 'unknown'
            key = f"{subject}|{predicate}|{c_type}"
            groups.setdefault(key, []).append(claim)

        return groups

    def validate_all_groups(self, claims: List[Any]) -> Dict[str, ValidationReport]:
        """
        对所有声明组进行交叉验证

        Args:
            claims: 所有声明

        Returns:
            每个组的ValidationReport字典
        """
        groups = self.group_claims_by_key(claims)
        results = {}
        for key, group_claims in groups.items():
            if len(group_claims) >= 1:
                results[key] = self.validate(group_claims)
        return results
