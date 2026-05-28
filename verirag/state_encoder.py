"""
State Encoder（状态编码器）

功能:
- 将query + documents + conflicts + history 编码为固定维度的state向量
- 输出328维state向量（用于RL Policy Network输入）
- 支持Query Encoder (BERT-based)、Document Encoder (Multi-head attention)、
  Conflict Encoder (MLP)、History Encoder (LSTM)

State维度分解:
- Query特征: 128维
- Document特征: 128维
- Conflict特征: 64维
- History特征: 8维
- 总计: 328维
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency guard
    AutoModel = None
    AutoTokenizer = None


class _TinyEncoder(nn.Module):
    """Container exposing a BERT-like `.layer` ModuleList."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(12)])


class _TinyTransformerBackbone(nn.Module):
    """
    Lightweight local fallback for offline/test environments.

    It mimics the subset of HuggingFace model output used by QueryEncoder:
    `.config.hidden_size`, `.embeddings`, `.encoder.layer`, and a forward
    return object containing `last_hidden_state`.
    """

    def __init__(self, vocab_size: int = 30522, hidden_size: int = 768):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder = _TinyEncoder(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Any:
        hidden = self.embeddings(input_ids.clamp(min=0, max=self.embeddings.num_embeddings - 1))
        # A single residual projection gives the fallback trainable capacity while
        # keeping tests and CPU smoke runs fast.
        if len(self.encoder.layer) > 0:
            hidden = hidden + 0.01 * self.encoder.layer[-1](hidden)
        hidden = self.layer_norm(hidden)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return SimpleNamespace(last_hidden_state=hidden)


class _SimpleTokenizer:
    """Small tokenizer fallback with a HuggingFace-like call interface."""

    vocab_size = 30522

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
        padding: bool = True,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else text
        token_rows = []
        mask_rows = []
        for item in texts:
            pieces = str(item).split()
            ids = [abs(hash(piece.lower())) % (self.vocab_size - 1) + 1 for piece in pieces]
            if truncation:
                ids = ids[:max_length]
            if padding:
                ids = ids + [0] * (max_length - len(ids))
            mask = [1 if token_id != 0 else 0 for token_id in ids]
            token_rows.append(ids)
            mask_rows.append(mask)
        return {
            "input_ids": torch.tensor(token_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        }


# ==================== Query Encoder ====================

class QueryEncoder(nn.Module):
    """
    查询编码器: 编码用户查询，输出query_risk_score和query_embedding

    Architecture: Pre-trained BERT + Risk Projection Head
    Output: [query_embedding(128), risk_score(1), adversarial_prob(1)]
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        num_risk_categories: int = 5,  # LOW, MEDIUM, HIGH, CRITICAL, ADVERSARIAL
        output_dim: int = 128,
        allow_remote_model_download: bool = False,
        use_pretrained_encoder: bool = False,
    ):
        super().__init__()

        # 加载预训练BERT；默认只使用本地缓存，缺失时回退到轻量本地编码器。
        self.using_pretrained_encoder = False
        if AutoModel is not None and AutoTokenizer is not None and use_pretrained_encoder:
            try:
                self.bert = AutoModel.from_pretrained(
                    bert_model_name,
                    local_files_only=not allow_remote_model_download,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    bert_model_name,
                    local_files_only=not allow_remote_model_download,
                )
                self.using_pretrained_encoder = True
            except Exception as exc:
                print(
                    "[StateEncoder] Pretrained query encoder unavailable; "
                    f"using lightweight fallback. reason={exc}"
                )
                self.bert = _TinyTransformerBackbone()
                self.tokenizer = _SimpleTokenizer()
        else:
            self.bert = _TinyTransformerBackbone()
            self.tokenizer = _SimpleTokenizer()

        # 冻结低层BERT参数以提高稳定性
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        # 冻结前8层（总共12层），只训练后4层
        for param in self.bert.encoder.layer[:8].parameters():
            param.requires_grad = False

        bert_dim = self.bert.config.hidden_size  # 768

        # Risk Score预测头（多任务）
        self.risk_classifier = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_risk_categories),
        )

        # Query Embedding投影（用于state fusion）
        self.embedding_proj = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # 对抗模式检测（keyword + semantic）
        self.adversarial_detector = nn.Sequential(
            nn.Linear(bert_dim + 20, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def extract_handcrafted_features(self, query_text: str) -> torch.Tensor:
        """
        手工特征提取（用于增强对抗检测）

        特征:
        - 特殊字符比例
        - 查询长度
        - 敏感关键词存在性
        - 大写字母比例
        - 数字比例
        """
        features = []
        text_len = max(len(query_text), 1)

        # 特殊字符比例
        special_chars = sum(1 for c in query_text if not c.isalnum() and not c.isspace())
        features.append(special_chars / text_len)

        # 查询长度（归一化）
        features.append(min(text_len / 500, 1.0))

        # 敏感关键词（one-hot，最多10个）
        sensitive_keywords = [
            'ignore previous', 'jailbreak', 'DAN', 'prompt injection',
            'ignore instructions', 'bypass', 'hack', 'exploit',
            'override', 'system prompt'
        ]
        keyword_matches = [
            1.0 if kw in query_text.lower() else 0.0
            for kw in sensitive_keywords[:10]
        ]
        features.extend(keyword_matches)

        # 大写字母比例
        upper_ratio = sum(1 for c in query_text if c.isupper()) / text_len
        features.append(upper_ratio)

        # 数字比例
        digit_ratio = sum(1 for c in query_text if c.isdigit()) / text_len
        features.append(digit_ratio)

        # 填充到20维
        while len(features) < 20:
            features.append(0.0)

        return torch.tensor(features[:20], dtype=torch.float32)

    def forward(self, query_tokens: Dict[str, torch.Tensor], query_text: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            query_tokens: {'input_ids': [B, L], 'attention_mask': [B, L]}
            query_text: 原始查询文本（用于手工特征提取，可选）

        Returns:
            {
                'embedding': [B, 128],
                'risk_logits': [B, 5],
                'risk_score': [B, 1],
                'adversarial_prob': [B, 1],
            }
        """
        # BERT编码
        bert_output = self.bert(**query_tokens)
        cls_token = bert_output.last_hidden_state[:, 0, :]  # [B, 768]

        # Risk分类
        risk_logits = self.risk_classifier(cls_token)  # [B, 5]
        risk_probs = F.softmax(risk_logits, dim=-1)

        # 连续risk score（加权求和）
        risk_weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0],
                                     device=risk_logits.device)
        risk_score = (risk_probs * risk_weights).sum(dim=-1, keepdim=True)  # [B, 1]

        # Embedding投影
        query_emb = self.embedding_proj(cls_token)  # [B, 128]

        # 对抗检测
        handcrafted = self._build_handcrafted_batch(query_text, cls_token)
        adv_input = torch.cat([cls_token, handcrafted], dim=-1)
        adv_prob = self.adversarial_detector(adv_input)  # [B, 1]

        return {
            'embedding': query_emb,
            'risk_logits': risk_logits,
            'risk_score': risk_score,
            'adversarial_prob': adv_prob,
        }

    def _build_handcrafted_batch(
        self,
        query_text: Optional[Union[str, List[str]]],
        cls_token: torch.Tensor,
    ) -> torch.Tensor:
        """Build a [B, 20] handcrafted feature tensor for adversarial detection."""
        batch_size = cls_token.size(0)
        device = cls_token.device

        if query_text is None:
            return torch.zeros(batch_size, 20, dtype=cls_token.dtype, device=device)

        if isinstance(query_text, str):
            rows = [self.extract_handcrafted_features(query_text) for _ in range(batch_size)]
        else:
            texts = list(query_text)
            if len(texts) < batch_size:
                texts.extend([""] * (batch_size - len(texts)))
            rows = [self.extract_handcrafted_features(texts[i]) for i in range(batch_size)]

        handcrafted = torch.stack(rows, dim=0).to(device=device, dtype=cls_token.dtype)
        return handcrafted

    def encode_text(self, query_text: str) -> Dict[str, torch.Tensor]:
        """便捷方法: 直接编码文本"""
        tokens = self.tokenizer(
            query_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        device = next(self.bert.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        return self.forward(tokens, query_text)


# ==================== Document Set Encoder ====================

class DocumentSetEncoder(nn.Module):
    """
    文档集合编码器: 编码检索到的文档集合

    Architecture: Individual Doc Encoding + Set Transformer
    Output: [doc_set_embedding(128), doc_stats(64)]
    """

    def __init__(
        self,
        bert_dim: int = 768,
        max_docs: int = 10,
        doc_embedding_dim: int = 256,
        num_heads: int = 8,
        num_set_layers: int = 2,
        output_dim: int = 128,
    ):
        super().__init__()
        self.max_docs = max_docs
        self.doc_embedding_dim = doc_embedding_dim
        self.output_dim = output_dim

        # 单个文档编码器（共享）
        self.doc_proj = nn.Sequential(
            nn.Linear(bert_dim, doc_embedding_dim),
            nn.LayerNorm(doc_embedding_dim),
            nn.GELU(),
        )

        # Set Transformer for cross-document interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=doc_embedding_dim,
            nhead=num_heads,
            dim_feedforward=doc_embedding_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.set_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_set_layers
        )

        # 文档统计编码器
        self.stats_encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(doc_embedding_dim + 64, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def compute_doc_statistics(
        self,
        doc_embeddings: torch.Tensor,  # [B, N, D]
        doc_scores: torch.Tensor,       # [B, N]
    ) -> torch.Tensor:
        """
        计算文档集合的统计特征

        特征:
        - 均值、方差、最大值、最小值
        - 分数分布熵
        - 两两相似度统计
        """
        B, N, D = doc_embeddings.shape
        if N == 0:
            return torch.zeros(B, 20, device=doc_embeddings.device, dtype=doc_embeddings.dtype)

        stats = []

        # 基本统计量
        stats.append(doc_embeddings.mean(dim=1))   # [B, D]
        stats.append(doc_embeddings.std(dim=1, unbiased=False))    # [B, D]
        stats.append(doc_embeddings.max(dim=1)[0]) # [B, D]
        stats.append(doc_embeddings.min(dim=1)[0]) # [B, D]

        # 检索分数统计
        stats.append(doc_scores.mean(dim=-1, keepdim=True))  # [B, 1]
        stats.append(doc_scores.std(dim=-1, keepdim=True, unbiased=False))   # [B, 1]
        stats.append(doc_scores.max(dim=-1, keepdim=True)[0]) # [B, 1]

        # 分数分布熵（多样性指标）
        score_probs = F.softmax(doc_scores, dim=-1)
        entropy = -(score_probs * torch.log(score_probs + 1e-8)).sum(dim=-1, keepdim=True)
        stats.append(entropy)  # [B, 1]

        # 两两余弦相似度统计
        doc_norm = F.normalize(doc_embeddings, dim=-1)
        sim_matrix = torch.bmm(doc_norm, doc_norm.transpose(1, 2))  # [B, N, N]
        mask = torch.eye(N, device=sim_matrix.device).bool().unsqueeze(0)
        sim_matrix_masked = sim_matrix.masked_fill(mask, float('nan'))

        mean_sim = torch.nan_to_num(sim_matrix_masked.nanmean(dim=[1, 2], keepdim=True), nan=0.0)
        stats.append(mean_sim)  # [B, 1]

        # 高相似度比例（潜在冲突指标）
        high_sim_ratio = torch.nan_to_num(
            (sim_matrix_masked > 0.9).float().nanmean(dim=[1, 2], keepdim=True),
            nan=0.0,
        )
        stats.append(high_sim_ratio)  # [B, 1]

        # 低相似度比例（来源多样性）
        low_sim_ratio = torch.nan_to_num(
            (sim_matrix_masked < 0.3).float().nanmean(dim=[1, 2], keepdim=True),
            nan=0.0,
        )
        stats.append(low_sim_ratio)  # [B, 1]

        # 拼接并截断到20维
        stats_concat = []
        for s in stats:
            if s.dim() == 2:
                stats_concat.append(s)
            elif s.shape[-1] == 1:
                stats_concat.append(s.squeeze(-1))

        result = torch.cat(stats_concat, dim=-1)
        return result[:, :20]

    def forward(
        self,
        doc_embeddings: torch.Tensor,   # [B, N, 768]
        doc_scores: torch.Tensor,        # [B, N]
        doc_masks: Optional[torch.Tensor] = None,  # [B, N]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            {
                'doc_set_embedding': [B, 128],
                'doc_stats': [B, 64],
                'individual_doc_emb': [B, N, 256],
            }
        """
        B, N, D = doc_embeddings.shape
        if N == 0:
            empty_docs = torch.zeros(B, 0, self.doc_embedding_dim, device=doc_embeddings.device)
            return {
                'doc_set_embedding': torch.zeros(B, self.output_dim, device=doc_embeddings.device),
                'doc_stats': torch.zeros(B, 64, device=doc_embeddings.device),
                'individual_doc_emb': empty_docs,
            }

        # 截断到max_docs
        if N > self.max_docs:
            doc_embeddings = doc_embeddings[:, :self.max_docs]
            doc_scores = doc_scores[:, :self.max_docs]
            if doc_masks is not None:
                doc_masks = doc_masks[:, :self.max_docs]
            N = self.max_docs

        # 单个文档投影
        doc_proj = self.doc_proj(doc_embeddings)  # [B, N, 256]

        # Set Transformer
        if doc_masks is not None:
            set_output = self.set_transformer(
                doc_proj,
                src_key_padding_mask=~doc_masks.bool()
            )
        else:
            set_output = self.set_transformer(doc_proj)

        # Mean pooling
        doc_set_emb = set_output.mean(dim=1)  # [B, 256]

        # 统计特征
        doc_stats_input = self.compute_doc_statistics(doc_embeddings, doc_scores)
        doc_stats = self.stats_encoder(doc_stats_input)  # [B, 64]

        # 融合输出
        combined = torch.cat([doc_set_emb, doc_stats], dim=-1)  # [B, 320]
        output = self.output_proj(combined)  # [B, 128]

        return {
            'doc_set_embedding': output,
            'doc_stats': doc_stats,
            'individual_doc_emb': doc_proj,
        }


# ==================== Conflict Encoder ====================

class ConflictEncoder(nn.Module):
    """
    冲突编码器: 量化检索文档之间的冲突和矛盾

    Output: [conflict_embedding(64), conflict_score(1), conflict_types(4)]
    """

    def __init__(
        self,
        doc_embedding_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()

        # 成对冲突检测器
        self.pairwise_conflict = nn.Sequential(
            nn.Linear(doc_embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),  # 4种冲突类型
        )

        # Self-attention over conflict pairs
        self.conflict_attention = nn.MultiheadAttention(
            embed_dim=4, num_heads=1, batch_first=True
        )

        # 冲突聚合
        self.conflict_aggregator = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, output_dim),
        )

        # 冲突严重性评分器
        self.severity_scorer = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def compute_pairwise_conflicts(
        self,
        doc_embeddings: torch.Tensor,  # [B, N, D]
    ) -> torch.Tensor:
        """
        计算所有文档对的冲突向量
        Returns: [B, num_pairs, 4]
        """
        B, N, D = doc_embeddings.shape

        if N < 2:
            return torch.zeros(B, 0, 4, device=doc_embeddings.device)

        # 生成所有对
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))

        num_pairs = len(pairs)

        # 构建pair embeddings
        pair_emb_list = []
        for i, j in pairs:
            pair_emb = torch.cat([
                doc_embeddings[:, i, :],
                doc_embeddings[:, j, :],
            ], dim=-1)  # [B, 2D]
            pair_emb_list.append(pair_emb)

        pair_embs = torch.stack(pair_emb_list, dim=1)  # [B, num_pairs, 2D]

        # 检测冲突
        pair_embs_flat = pair_embs.reshape(B * num_pairs, -1)
        conflict_logits = self.pairwise_conflict(pair_embs_flat)
        conflict_probs = torch.sigmoid(conflict_logits)
        conflict_probs = conflict_probs.view(B, num_pairs, 4)

        return conflict_probs

    def forward(
        self,
        doc_embeddings: torch.Tensor,  # [B, N, D]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            {
                'conflict_embedding': [B, 64],
                'conflict_score': [B, 1],
                'conflict_types': [B, 4],
            }
        """
        B, N, D = doc_embeddings.shape

        if N < 2:
            return {
                'conflict_embedding': torch.zeros(B, 64, device=doc_embeddings.device),
                'conflict_score': torch.zeros(B, 1, device=doc_embeddings.device),
                'conflict_types': torch.zeros(B, 4, device=doc_embeddings.device),
            }

        # 计算成对冲突
        conflict_matrix = self.compute_pairwise_conflicts(doc_embeddings)

        if conflict_matrix.size(1) == 0:
            return {
                'conflict_embedding': torch.zeros(B, 64, device=doc_embeddings.device),
                'conflict_score': torch.zeros(B, 1, device=doc_embeddings.device),
                'conflict_types': torch.zeros(B, 4, device=doc_embeddings.device),
            }

        # Self-attention聚合
        conflict_attended, _ = self.conflict_attention(
            conflict_matrix, conflict_matrix, conflict_matrix
        )

        # 跨pairs聚合
        conflict_agg = conflict_attended.mean(dim=1)  # [B, 4]

        # Per-type scores
        conflict_types = conflict_agg  # [B, 4]

        # Embed conflict vector
        conflict_embedding = self.conflict_aggregator(conflict_agg)  # [B, 64]

        # Overall severity
        conflict_score = self.severity_scorer(conflict_embedding)  # [B, 1]

        return {
            'conflict_embedding': conflict_embedding,
            'conflict_score': conflict_score,
            'conflict_types': conflict_types,
        }


# ==================== History Encoder ====================

class HistoryEncoder(nn.Module):
    """
    历史验证模式编码器: 编码过去K步的验证决策历史

    Architecture: LSTM + Temporal Attention
    Output: [history_embedding(64)]
    """

    def __init__(
        self,
        num_actions: int = 5,
        history_length: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()
        self.history_length = history_length
        self.num_actions = num_actions

        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, 16)

        # Result embedding (success/failure/unknown)
        self.result_embedding = nn.Embedding(3, 8)

        # LSTM序列编码
        input_dim = 16 + 8 + 8  # action + result + time = 32
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def compute_time_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        计算时间特征（位置编码风格）
        timestamps: [B, K] relative time steps
        """
        B, K = timestamps.shape
        device = timestamps.device

        position = timestamps.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, 8, 2, device=device).float() *
            (-torch.log(torch.tensor(10000.0)) / 8)
        )

        pe = torch.zeros(B, K, 8, device=device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        action_history: torch.Tensor,    # [B, K]
        result_history: torch.Tensor,    # [B, K]
        valid_mask: Optional[torch.Tensor] = None,  # [B, K]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            {
                'history_embedding': [B, 64],
                'history_attention': [B, K],
            }
        """
        B, K = action_history.shape
        device = action_history.device

        if K == 0:
            return {
                'history_embedding': torch.zeros(B, 64, device=device),
                'history_attention': torch.ones(B, 1, device=device) / B,
            }

        # Embeddings
        action_emb = self.action_embedding(action_history.clamp(0, self.num_actions - 1))
        result_emb = self.result_embedding(result_history.clamp(0, 2))

        # 时间特征
        timestamps = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
        timestamps = K - 1 - timestamps  # Reverse: most recent = 0
        time_emb = self.compute_time_features(timestamps)

        # 拼接特征
        history_input = torch.cat([action_emb, result_emb, time_emb], dim=-1)

        # LSTM编码
        if valid_mask is not None:
            lengths = valid_mask.sum(dim=1).clamp(min=1).cpu().long()
            packed = nn.utils.rnn.pack_padded_sequence(
                history_input, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_output, _ = self.lstm(packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        else:
            lstm_output, _ = self.lstm(history_input)

        # Temporal attention
        attn_scores = self.temporal_attention(lstm_output).squeeze(-1)

        if valid_mask is not None:
            attn_scores = attn_scores.masked_fill(~valid_mask.bool(), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权聚合
        history_embedding = torch.bmm(
            attn_weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)

        # 输出投影
        history_embedding = self.output_proj(history_embedding)

        return {
            'history_embedding': history_embedding,
            'history_attention': attn_weights,
        }


# ==================== State Fusion Module ====================

class StateFusionModule(nn.Module):
    """
    状态融合模块: 将所有编码器输出融合为统一的state表示

    使用Cross-Attention + Gated Fusion机制
    """

    def __init__(
        self,
        query_dim: int = 128,
        doc_dim: int = 128,
        conflict_dim: int = 64,
        history_dim: int = 64,
        output_dim: int = 512,
    ):
        super().__init__()

        total_dim = query_dim + doc_dim + conflict_dim + history_dim

        # Gated fusion
        self.gate_proj = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Linear(total_dim // 2, 4),
            nn.Softmax(dim=-1),
        )

        # 投影层
        self.query_proj = nn.Linear(query_dim, output_dim)
        self.doc_proj = nn.Linear(doc_dim, output_dim)
        self.conflict_proj = nn.Linear(conflict_dim, output_dim)
        self.history_proj = nn.Linear(history_dim, output_dim)

        # 最终融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        query_emb: torch.Tensor,    # [B, 128]
        doc_emb: torch.Tensor,      # [B, 128]
        conflict_emb: torch.Tensor, # [B, 64]
        history_emb: torch.Tensor,  # [B, 64]
    ) -> torch.Tensor:
        """
        融合所有特征

        Returns: [B, 512]
        """
        concat = torch.cat([query_emb, doc_emb, conflict_emb, history_emb], dim=-1)
        gates = self.gate_proj(concat)  # [B, 4]

        # 投影
        q_proj = self.query_proj(query_emb)
        d_proj = self.doc_proj(doc_emb)
        c_proj = self.conflict_proj(conflict_emb)
        h_proj = self.history_proj(history_emb)

        # 应用gate
        q_gated = q_proj * gates[:, 0:1]
        d_gated = d_proj * gates[:, 1:2]
        c_gated = c_proj * gates[:, 2:3]
        h_gated = h_proj * gates[:, 3:4]

        # 拼接融合
        fused_input = torch.cat([q_gated, d_gated, c_gated, h_gated], dim=-1)
        state = self.fusion_mlp(fused_input)

        return state


# ==================== Hierarchical Policy Head ====================

class HierarchicalPolicyHead(nn.Module):
    """
    层次化策略头: 支持两级决策
    Level 1: {SKIP, VERIFY, REJECT}
    Level 2 (if VERIFY): {LIGHT, DEEP, EXPAND}

    输出对应 action space: {SKIP, LIGHT, DEEP, EXPAND, REJECT} = 5个动作
    """

    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 256,
        num_actions: int = 5,
    ):
        super().__init__()

        # Level 1: SKIP vs VERIFY vs REJECT
        self.level1_logits = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # [SKIP, VERIFY, REJECT]
        )

        # Level 2: Verification type (only if VERIFY)
        self.level2_logits = nn.Sequential(
            nn.Linear(state_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # [LIGHT, DEEP, EXPAND]
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            {
                'action_logits': [B, 5],
                'action_probs': [B, 5],
                'value': [B, 1],
                'level1_probs': [B, 3],
                'level2_probs': [B, 3],
            }
        """
        # Level 1 decision
        l1_logits = self.level1_logits(state)  # [B, 3]
        l1_probs = F.softmax(l1_logits / max(temperature, 1e-6), dim=-1)

        # Level 2 decision
        l2_input = torch.cat([state, l1_probs], dim=-1)
        l2_logits = self.level2_logits(l2_input)  # [B, 3]
        l2_probs = F.softmax(l2_logits / max(temperature, 1e-6), dim=-1)

        # 组合为完整action space
        # SKIP (0) = l1_probs[:, 0]
        # LIGHT (1) = l1_probs[:, 1] * l2_probs[:, 0]
        # DEEP (2) = l1_probs[:, 1] * l2_probs[:, 1]
        # EXPAND (3) = l1_probs[:, 1] * l2_probs[:, 2]
        # REJECT (4) = l1_probs[:, 2]

        B = state.shape[0]
        action_probs = torch.zeros(B, 5, device=state.device)
        action_probs[:, 0] = l1_probs[:, 0]      # SKIP
        action_probs[:, 1] = l1_probs[:, 1] * l2_probs[:, 0]  # LIGHT
        action_probs[:, 2] = l1_probs[:, 1] * l2_probs[:, 1]  # DEEP
        action_probs[:, 3] = l1_probs[:, 1] * l2_probs[:, 2]  # EXPAND
        action_probs[:, 4] = l1_probs[:, 2]      # REJECT

        # 重新归一化
        action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Logits for PPO
        action_logits = torch.log(action_probs + 1e-8)

        # Value estimate
        value = self.value_head(state)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'value': value,
            'level1_probs': l1_probs,
            'level2_probs': l2_probs,
        }


# ==================== State Encoder（完整封装） ====================

class StateEncoder(nn.Module):
    """
    完整的状态编码器

    整合所有子编码器，输出328维state向量
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        max_docs: int = 10,
        history_length: int = 10,
        num_actions: int = 5,
        output_dim: int = 512,
        allow_remote_model_download: bool = False,
        use_pretrained_encoder: bool = False,
    ):
        super().__init__()

        self.query_encoder = QueryEncoder(
            bert_model_name=bert_model_name,
            output_dim=128,
            allow_remote_model_download=allow_remote_model_download,
            use_pretrained_encoder=use_pretrained_encoder,
        )
        self.doc_encoder = DocumentSetEncoder(
            max_docs=max_docs,
            output_dim=128,
        )
        self.conflict_encoder = ConflictEncoder(
            output_dim=64,
        )
        self.history_encoder = HistoryEncoder(
            history_length=history_length,
            num_actions=num_actions,
            output_dim=64,
        )
        self.fusion = StateFusionModule(
            query_dim=128,
            doc_dim=128,
            conflict_dim=64,
            history_dim=64,
            output_dim=output_dim,
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        编码完整状态

        Args:
            inputs: {
                'query_tokens': {'input_ids': [B, L], 'attention_mask': [B, L]},
                'query_text': str (可选),
                'doc_embeddings': [B, N, 768],
                'doc_scores': [B, N],
                'doc_masks': [B, N] (可选),
                'action_history': [B, K] (可选),
                'result_history': [B, K] (可选),
                'history_mask': [B, K] (可选),
            }

        Returns:
            {
                'state': [B, 512],
                'query_emb': [B, 128],
                'doc_emb': [B, 128],
                'conflict_emb': [B, 64],
                'history_emb': [B, 64],
                'risk_score': [B, 1],
                'adversarial_prob': [B, 1],
                'conflict_score': [B, 1],
            }
        """
        # Query编码
        query_out = self.query_encoder(
            inputs['query_tokens'],
            inputs.get('query_text')
        )

        # Document编码
        doc_out = self.doc_encoder(
            inputs['doc_embeddings'],
            inputs['doc_scores'],
            inputs.get('doc_masks')
        )

        # Conflict编码
        conflict_out = self.conflict_encoder(doc_out['individual_doc_emb'])

        # History编码
        if 'action_history' in inputs and inputs['action_history'].size(1) > 0:
            history_out = self.history_encoder(
                inputs['action_history'],
                inputs.get('result_history', torch.zeros_like(inputs['action_history'])),
                inputs.get('history_mask')
            )
        else:
            B = inputs['doc_embeddings'].size(0)
            device = inputs['doc_embeddings'].device
            history_out = {
                'history_embedding': torch.zeros(B, 64, device=device),
                'history_attention': torch.ones(B, 1, device=device) / B,
            }

        # 状态融合
        state = self.fusion(
            query_out['embedding'],
            doc_out['doc_set_embedding'],
            conflict_out['conflict_embedding'],
            history_out['history_embedding'],
        )

        return {
            'state': state,
            'query_emb': query_out['embedding'],
            'doc_emb': doc_out['doc_set_embedding'],
            'conflict_emb': conflict_out['conflict_embedding'],
            'history_emb': history_out['history_embedding'],
            'risk_score': query_out['risk_score'],
            'adversarial_prob': query_out['adversarial_prob'],
            'conflict_score': conflict_out['conflict_score'],
        }

    def encode_query(self, query_text: str) -> Dict[str, torch.Tensor]:
        """便捷方法: 只编码查询"""
        return self.query_encoder.encode_text(query_text)
