"""
Module 1: Claim Extractor（声明提取器）

功能:
- 从文档中提取结构化声明（实体、数值、关系）
- 混合提取策略: 规则引擎（数值/时间）+ NER + 轻量LLM
- 防御Embedding Blind Spot: 保留原始数值字符串

设计要点:
1. 规则引擎对数值/时间100%精确
2. LLM覆盖复杂关系/因果声明
3. 始终保留原始文本用于精确比对
4. 查询感知: 根据query类型调整提取优先级
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum, auto

import torch
import torch.nn as nn


# ==================== 数据类型定义 ====================

class ClaimType(Enum):
    """声明类型枚举"""
    FACTUAL = "factual"       # 事实性声明（实体、关系）
    NUMERICAL = "numerical"   # 数值性声明（精确数值、范围）
    TEMPORAL = "temporal"     # 时间性声明（日期、时间段）
    CAUSAL = "causal"         # 因果性声明（原因->结果）
    COMPARATIVE = "comparative"  # 比较性声明（A > B, A = B）
    ATTRIBUTIVE = "attributive"  # 归因性声明（归因于...）


@dataclass
class DocSpan:
    """文档中的来源位置"""
    start_char: int
    end_char: int
    text_snippet: str


@dataclass
class Claim:
    """
    结构化声明 - Claim Extractor的核心输出

    关键设计: value字段保存原始字符串（不经过embedding）
    这是防御Embedding Blind Spot的核心机制
    """
    # 唯一标识
    claim_id: str = ""                           # 唯一标识符 (hash)
    claim_type: ClaimType = ClaimType.FACTUAL    # 声明类型

    # 核心三元组
    subject: str = ""                            # 主语实体
    predicate: str = ""                          # 谓语/关系
    object: str = ""                             # 宾语实体

    # 数值相关（防御Embedding Blind Spot的关键）
    value: Optional[str] = None                  # 原始数值字符串！不经过embedding
    numeric_precision: Optional[int] = None      # 数值精度（小数位数）
    value_range: Optional[Tuple[float, float]] = None  # 数值范围

    # 时间约束
    temporal_scope: Optional[Tuple[str, str]] = None  # 时间范围 [start, end]

    # 来源追踪
    doc_id: str = ""                             # 来源文档ID
    source_text_span: Tuple[int, int] = (0, 0)   # 在原文中的位置
    raw_text: str = ""                           # 原始文本片段

    # 元信息
    confidence: float = 0.0                      # 提取置信度 [0,1]
    extraction_method: str = ""                  # 提取方法（rule/llm/hybrid）
    query_relevance: float = 0.0                 # 与查询的相关性


@dataclass
class ClaimExtractionResult:
    """单个文档的声明提取结果"""
    doc_id: str = ""
    claims: List[Claim] = field(default_factory=list)
    extraction_time_ms: float = 0.0
    failed_spans: List[str] = field(default_factory=list)


# ==================== 数值模式定义 ====================

# 数值正则表达式模式（高精度匹配）
NUMERICAL_PATTERNS = {
    # 货币格式: $15K, $1.5M, $10,000.00
    'currency': re.compile(
        r'[\$\€\£\¥]\s*\d+(?:[,.]\d+)?\s*[KMBT]?|\d+(?:[,.]\d+)?\s*[KMBT]?\s*(?:USD|EUR|GBP|JPY|dollars?|euros?|pounds?)',
        re.IGNORECASE
    ),
    # 百分比: 15%, 15.5%, 15 percent
    'percentage': re.compile(
        r'\d+(?:\.\d+)?\s*(?:%|percent|percentage)',
        re.IGNORECASE
    ),
    # 纯数字: 1234, 1,234,567, 3.14159
    'number': re.compile(
        r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d+(?:\.\d+)?\b'
    ),
    # 范围: 10-20, 10~20, between 10 and 20
    'range': re.compile(
        r'(?:between\s+)?(\d+(?:\.\d+)?)\s*(?:-|~|to|and)\s*(\d+(?:\.\d+)?)',
        re.IGNORECASE
    ),
}

# 时间正则表达式模式
TEMPORAL_PATTERNS = {
    # ISO日期: 2024-01-15, 2024/01/15
    'iso_date': re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'),
    # 月份日期: January 15, 2024; Jan 15, 2024
    'month_date': re.compile(
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?[,\s]+\d{4}',
        re.IGNORECASE
    ),
    # 年份: 2024, in 2024, since 2024
    'year': re.compile(r'\b(?:in|since|from|until|before|after)\s+(\d{4})|\b(\d{4})\b'),
    # 时间段: from 2020 to 2024, between 2020 and 2024
    'time_range': re.compile(
        r'(?:from|between)\s+(\d{4})\s+(?:to|and)\s+(\d{4})',
        re.IGNORECASE
    ),
}

# 实体-关系模式（主谓宾提取）
ENTITY_RELATION_PATTERNS = [
    # PATTERN: "X is Y" -> subject=X, predicate=is_a, object=Y
    re.compile(r'\b([^,\.]{3,50})\s+is\s+(?:a|an|the)\s+([^,\.]{3,50})\b', re.IGNORECASE),
    # PATTERN: "X was born in Y" -> subject=X, predicate=born_in, object=Y
    re.compile(r'\b([^,\.]{3,50})\s+was\s+born\s+in\s+([^,\.]{3,50})\b', re.IGNORECASE),
    # PATTERN: "X founded Y" -> subject=X, predicate=founded, object=Y
    re.compile(r'\b([^,\.]{3,50})\s+founded\s+([^,\.]{3,50})\b', re.IGNORECASE),
    # PATTERN: "X has Y" -> subject=X, predicate=has, object=Y
    re.compile(r'\b([^,\.]{3,50})\s+has\s+(?:a|an|the)?\s*([^,\.]{3,50})\b', re.IGNORECASE),
]


# ==================== Claim Extractor 主类 ====================

class ClaimExtractor(nn.Module):
    """
    声明提取器: 从文档中提取结构化声明

    混合提取策略:
    - 规则引擎: 数值/时间/实体关系（100%精确，无LLM调用开销）
    - NER模型: 命名实体识别（辅助实体提取）
    - 轻量LLM: 复杂声明（因果/比较，可选）

    输出: Claim对象列表，每个Claim包含原始字符串value（不经过embedding）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化声明提取器

        Args:
            config: 配置字典
                - rule_engine_enabled: bool (默认True) - 启用规则引擎
                - llm_extractor_enabled: bool (默认False) - 启用LLM提取器
                - min_confidence: float (默认0.7) - 最低置信度阈值
                - extract_all_numerical: bool (默认True) - 提取所有数值
                - preserve_original_string: bool (默认True) - 保留原始字符串
        """
        super().__init__()
        self.config = config or {}
        self.rule_engine_enabled = self.config.get('rule_engine_enabled', True)
        self.llm_extractor_enabled = self.config.get('llm_extractor_enabled', False)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.extract_all_numerical = self.config.get('extract_all_numerical', True)
        self.preserve_original_string = self.config.get('preserve_original_string', True)

        # 初始化轻量LLM（用于复杂声明提取，可选）
        self.llm_model = None
        if self.llm_extractor_enabled:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                llm_name = self.config.get('llm_model', 'microsoft/phi-2')
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map='auto' if torch.cuda.is_available() else None,
                )
                self.llm_model.eval()
            except Exception as e:
                print(f"[ClaimExtractor] LLM加载失败，仅使用规则引擎: {e}")
                self.llm_extractor_enabled = False

    def extract(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> List[Claim]:
        """
        从文档列表中提取结构化声明

        Args:
            documents: 文档文本列表
            doc_ids: 文档ID列表（可选，自动生成）

        Returns:
            Claim对象列表
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        all_claims = []
        for doc_text, doc_id in zip(documents, doc_ids):
            doc_claims = self._extract_from_single_doc(doc_text, doc_id)
            all_claims.extend(doc_claims)

        return all_claims

    def extract_with_result(self, documents: List[Dict[str, Any]]) -> List[ClaimExtractionResult]:
        """
        从文档字典列表中提取声明，返回完整结果

        Args:
            documents: 文档字典列表，每项包含 {doc_id, text, metadata}

        Returns:
            ClaimExtractionResult列表
        """
        import time
        results = []
        for doc in documents:
            start_time = time.time()
            doc_id = doc.get('doc_id', doc.get('id', 'unknown'))
            doc_text = doc.get('text', doc.get('content', ''))

            claims = self._extract_from_single_doc(doc_text, doc_id)
            elapsed_ms = (time.time() - start_time) * 1000

            result = ClaimExtractionResult(
                doc_id=doc_id,
                claims=claims,
                extraction_time_ms=elapsed_ms,
            )
            results.append(result)

        return results

    def _extract_from_single_doc(self, doc_text: str, doc_id: str) -> List[Claim]:
        """
        从单个文档提取声明（内部方法）

        执行流程:
        1. 规则引擎提取（数值/时间/实体关系）
        2. LLM辅助提取（复杂声明，可选）
        3. 融合去重
        4. 置信度过滤
        """
        claims = []

        # Step 1: 规则引擎提取
        if self.rule_engine_enabled:
            claims.extend(self._rule_based_extraction(doc_text, doc_id))

        # Step 2: LLM辅助提取（可选）
        if self.llm_extractor_enabled and self.llm_model is not None:
            llm_claims = self._llm_based_extraction(doc_text, doc_id)
            claims.extend(llm_claims)

        # Step 3: 去重（按subject+predicate+object）
        claims = self._deduplicate_claims(claims)

        # Step 4: 置信度过滤
        claims = [c for c in claims if c.confidence >= self.min_confidence]

        return claims

    def _rule_based_extraction(self, doc_text: str, doc_id: str) -> List[Claim]:
        """
        基于规则的声明提取

        提取策略:
        - 数值模式: 货币、百分比、纯数字、范围
        - 时间模式: ISO日期、月份日期、年份、时间段
        - 实体关系: 主谓宾模式匹配
        """
        claims = []

        # 1. 数值提取
        if self.extract_all_numerical:
            for num_type, pattern in NUMERICAL_PATTERNS.items():
                for match in pattern.finditer(doc_text):
                    claim = self._parse_numerical_claim(match, num_type, doc_text, doc_id)
                    if claim:
                        claims.append(claim)

        # 2. 时间提取
        for temp_type, pattern in TEMPORAL_PATTERNS.items():
            for match in pattern.finditer(doc_text):
                claim = self._parse_temporal_claim(match, temp_type, doc_text, doc_id)
                if claim:
                    claims.append(claim)

        # 3. 实体关系提取
        for pattern in ENTITY_RELATION_PATTERNS:
            for match in pattern.finditer(doc_text):
                claim = self._parse_entity_relation_claim(match, doc_text, doc_id)
                if claim:
                    claims.append(claim)

        return claims

    def _parse_numerical_claim(
        self, match: re.Match, num_type: str,
        doc_text: str, doc_id: str
    ) -> Optional[Claim]:
        """解析数值匹配为Claim对象"""
        raw_str = match.group(0)
        start, end = match.span()

        # 提取数字部分
        number_match = re.search(r'\d+(?:[,.]\d+)?', raw_str.replace(',', ''))
        if not number_match:
            return None

        try:
            numeric_value = float(number_match.group(0).replace(',', ''))
            # 计算精度（小数位数）
            decimal_part = number_match.group(0).split('.')
            precision = len(decimal_part[1]) if len(decimal_part) > 1 else 0
        except ValueError:
            numeric_value = None
            precision = None

        # 生成claim_id
        claim_id = self._generate_claim_id(doc_id, raw_str, start)

        # 构建subject（数值前面的上下文）
        context_start = max(0, start - 50)
        context = doc_text[context_start:start].strip()
        # 提取最后一个名词短语作为subject
        subject = self._extract_last_noun_phrase(context) or "unknown"

        # 确定predicate
        predicate_map = {
            'currency': 'has_value',
            'percentage': 'has_percentage',
            'number': 'has_number',
            'range': 'has_range',
        }

        claim = Claim(
            claim_id=claim_id,
            claim_type=ClaimType.NUMERICAL,
            subject=subject,
            predicate=predicate_map.get(num_type, 'has_value'),
            object=raw_str,
            value=raw_str if self.preserve_original_string else str(numeric_value),
            numeric_precision=precision,
            doc_id=doc_id,
            source_text_span=(start, end),
            raw_text=doc_text[max(0, start-20):min(len(doc_text), end+20)],
            confidence=0.9,  # 规则引擎置信度较高
            extraction_method='rule_engine',
        )
        return claim

    def _parse_temporal_claim(
        self, match: re.Match, temp_type: str,
        doc_text: str, doc_id: str
    ) -> Optional[Claim]:
        """解析时间匹配为Claim对象"""
        raw_str = match.group(0)
        start, end = match.span()

        claim_id = self._generate_claim_id(doc_id, raw_str, start)

        # 提取上下文
        context_start = max(0, start - 50)
        context = doc_text[context_start:start].strip()
        subject = self._extract_last_noun_phrase(context) or "unknown"

        # 确定predicate
        predicate_map = {
            'iso_date': 'date_is',
            'month_date': 'date_is',
            'year': 'year_is',
            'time_range': 'time_range_is',
        }

        claim = Claim(
            claim_id=claim_id,
            claim_type=ClaimType.TEMPORAL,
            subject=subject,
            predicate=predicate_map.get(temp_type, 'date_is'),
            object=raw_str,
            value=raw_str,  # 保留原始时间字符串
            doc_id=doc_id,
            source_text_span=(start, end),
            raw_text=doc_text[max(0, start-20):min(len(doc_text), end+20)],
            confidence=0.85,
            extraction_method='rule_engine',
        )
        return claim

    def _parse_entity_relation_claim(
        self, match: re.Match, doc_text: str, doc_id: str
    ) -> Optional[Claim]:
        """解析实体关系匹配为Claim对象"""
        groups = match.groups()
        if len(groups) < 2:
            return None

        subject = groups[0].strip()
        object_val = groups[1].strip()
        raw_str = match.group(0)
        start, end = match.span()

        # 推断predicate
        predicate = self._infer_predicate(raw_str)

        claim_id = self._generate_claim_id(doc_id, raw_str, start)

        claim = Claim(
            claim_id=claim_id,
            claim_type=ClaimType.FACTUAL,
            subject=subject,
            predicate=predicate,
            object=object_val,
            value=None,
            doc_id=doc_id,
            source_text_span=(start, end),
            raw_text=raw_str,
            confidence=0.75,  # 模式匹配置信度稍低
            extraction_method='rule_engine',
        )
        return claim

    def _llm_based_extraction(self, doc_text: str, doc_id: str) -> List[Claim]:
        """
        基于轻量LLM的声明提取（处理复杂声明）

        用于提取:
        - 因果性声明（A导致了B）
        - 比较性声明（A比B更大）
        - 复杂关系（需要推理的声明）
        """
        if self.llm_model is None or self.llm_tokenizer is None:
            return []

        claims = []
        try:
            # 构造prompt
            prompt = self._build_extraction_prompt(doc_text)

            # 编码并生成
            inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                )

            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 解析LLM输出（期望JSON格式）
            parsed_claims = self._parse_llm_output(response, doc_text, doc_id)
            claims.extend(parsed_claims)

        except Exception as e:
            print(f"[ClaimExtractor] LLM提取失败: {e}")

        return claims

    def _build_extraction_prompt(self, doc_text: str) -> str:
        """构建LLM提取prompt"""
        truncated_text = doc_text[:1500]  # 截断以避免过长
        prompt = (
            "Extract all factual claims from the following text. "
            "For each claim, identify: subject, predicate, object, and claim_type "
            "(one of: factual, numerical, temporal, causal, comparative).\n\n"
            f"Text: {truncated_text}\n\n"
            "Claims (JSON format):"
        )
        return prompt

    def _parse_llm_output(self, response: str, doc_text: str, doc_id: str) -> List[Claim]:
        """解析LLM输出为Claim列表（简化版，实际可用JSON解析）"""
        claims = []
        # 简化处理: 从文本中提取结构化信息
        # 实际实现中应使用更健壮的JSON解析
        return claims

    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """按subject+predicate+object去重"""
        seen = set()
        unique_claims = []
        for claim in claims:
            key = f"{claim.subject.lower()}|{claim.predicate.lower()}|{claim.object.lower()}"
            if key not in seen:
                seen.add(key)
                unique_claims.append(claim)
        return unique_claims

    @staticmethod
    def _generate_claim_id(doc_id: str, raw_str: str, position: int) -> str:
        """生成确定性claim ID"""
        content = f"{doc_id}:{raw_str}:{position}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @staticmethod
    def _extract_last_noun_phrase(context: str) -> Optional[str]:
        """
        从上下文中提取最后一个名词短语（简化版）

        实际实现可以使用spaCy NER进行更精确的提取
        """
        # 简单的启发式: 提取最后3-5个单词
        words = context.split()
        if len(words) >= 2:
            return ' '.join(words[-3:]).strip('.,;')
        return None

    @staticmethod
    def _infer_predicate(sentence: str) -> str:
        """
        从句子中推断predicate

        使用关键词匹配推断关系类型
        """
        sentence_lower = sentence.lower()

        # 因果关系
        causal_keywords = ['cause', 'lead', 'result', 'because', 'due to', 'resulted in',
                          'led to', 'caused by', 'because of']
        for kw in causal_keywords:
            if kw in sentence_lower:
                return 'causes'

        # 比较关系
        comparative_keywords = ['more than', 'less than', 'larger', 'smaller',
                               'better', 'worse', 'earlier', 'later', 'superior']
        for kw in comparative_keywords:
            if kw in sentence_lower:
                return 'comparative_relation'

        # 归属关系
        if 'founded' in sentence_lower or 'created' in sentence_lower:
            return 'founded'
        if 'born' in sentence_lower:
            return 'born_in'
        if 'is a' in sentence_lower or 'is an' in sentence_lower:
            return 'is_a'

        # 默认
        return 'has_relation'

    def forward(self, document_texts: List[str]) -> List[List[Claim]]:
        """
        PyTorch forward接口（用于与神经网络管道集成）

        Args:
            document_texts: 文档文本列表

        Returns:
            每个文档的Claim列表的列表
        """
        results = []
        for i, text in enumerate(document_texts):
            claims = self._extract_from_single_doc(text, f"batch_doc_{i}")
            results.append(claims)
        return results
