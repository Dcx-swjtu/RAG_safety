"""
Module 3: Verification Policy Network（RL策略网络）

架构:
- State Encoder: 编码query+docs+conflicts+history -> 512维state
- Hierarchical Policy Head: 两级决策 -> 5个action
- Value Head: 估计state value

Action Space (5个离散动作):
- 0: KEEP_DOCS - 保留当前文档集合
- 1: DROP_SUSPECT_DOCS - 删除高风险文档
- 2: RERANK_DOCS - 按文档风险/支持度重排
- 3: DROP_AND_RERANK - 删除高风险文档并重排剩余证据
- 4: ABSTAIN - 证据风险过高时拒答/不作答

训练: PPO (Proximal Policy Optimization)
"""

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_encoder import StateEncoder, HierarchicalPolicyHead


class VerificationPolicyNetwork(nn.Module):
    """
    完整的验证策略网络

    包含:
    - StateEncoder: 编码所有输入为state向量
    - HierarchicalPolicyHead: 输出action概率和state value
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化策略网络

        Args:
            config: 配置字典
                - bert_model_name: str (默认"bert-base-uncased")
                - max_docs: int (默认10)
                - history_length: int (默认10)
                - num_actions: int (默认5)
                - state_dim: int (默认512)
                - hidden_dim: int (默认256)
        """
        super().__init__()
        self.config = config or {}

        self.state_dim = self.config.get('state_dim', 512)
        self.num_actions = self.config.get('num_actions', self.config.get('action_dim', 5))

        # State Encoder
        self.state_encoder = StateEncoder(
            bert_model_name=self.config.get('bert_model_name', 'bert-base-uncased'),
            max_docs=self.config.get('max_docs', 10),
            history_length=self.config.get('history_length', 10),
            num_actions=self.num_actions,
            output_dim=self.state_dim,
            allow_remote_model_download=self.config.get('allow_remote_model_download', False),
            use_pretrained_encoder=self.config.get('use_pretrained_encoder', False),
        )

        # Policy Head (与StateEncoder内部共享)
        self.policy_head = HierarchicalPolicyHead(
            state_dim=self.state_dim,
            hidden_dim=self.config.get('hidden_dim', 256),
            num_actions=self.num_actions,
        )

        # 动作到名称映射
        self.action_names = {
            0: 'KEEP_DOCS',
            1: 'DROP_SUSPECT_DOCS',
            2: 'RERANK_DOCS',
            3: 'DROP_AND_RERANK',
            4: 'ABSTAIN',
        }

        # Some callers wrap the module with DataParallel directly. PyTorch
        # requires parameters to already be on cuda:0 in that case.
        if (
            self.config.get('auto_move_to_cuda_for_dataparallel', True)
            and torch.cuda.device_count() > 1
            and next(self.parameters()).device.type == 'cpu'
        ):
            self.cuda(0)

    def forward(
        self,
        inputs: Dict[str, Any],
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            inputs: 包含query_tokens, doc_embeddings等的字典
            temperature: 采样温度（高=更多探索，低=更确定）

        Returns:
            完整输出字典，包含:
            - state: [B, 512]
            - action_logits: [B, 5]
            - action_probs: [B, 5]
            - value: [B, 1]
            - level1/level2_probs: 层次决策概率
        """
        # 编码state
        encoder_out = self.state_encoder(inputs)
        state = encoder_out['state']

        # Policy输出
        policy_out = self.policy_head(state, temperature)

        return {
            'state': state,
            **policy_out,
            'risk_score': encoder_out['risk_score'],
            'adversarial_prob': encoder_out['adversarial_prob'],
            'conflict_score': encoder_out['conflict_score'],
        }

    def select_action(
        self,
        inputs: Dict[str, Any],
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        选择动作（推理时使用）

        Args:
            inputs: 输入字典
            deterministic: 是否确定性选择（True=取argmax）
            temperature: 采样温度

        Returns:
            {
                'action': int,
                'action_name': str,
                'action_probs': [5],
                'value': float,
                'log_prob': float,
            }
        """
        with torch.no_grad():
            output = self.forward(inputs, temperature)

        action_probs = output['action_probs'][0].cpu().numpy()
        value = output['value'][0].item()

        if deterministic:
            action = int(action_probs.argmax())
        else:
            action = int(torch.multinomial(output['action_probs'][0], 1).item())

        log_prob = torch.log(output['action_probs'][0, action] + 1e-8).item()

        return {
            'action': action,
            'action_name': self.action_names[action],
            'action_probs': action_probs,
            'value': value,
            'log_prob': log_prob,
            'state': output['state'],
            'risk_score': output['risk_score'][0].item(),
            'conflict_score': output['conflict_score'][0].item(),
        }

    def evaluate_actions(
        self,
        inputs: Dict[str, Any],
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        评估给定动作的log_prob和value（PPO更新时使用）

        Args:
            inputs: 输入字典
            actions: [B] 选中的动作索引

        Returns:
            {
                'log_probs': [B],
                'values': [B],
                'entropy': [B],
            }
        """
        if 'state' in inputs and 'query_tokens' not in inputs:
            state = inputs['state']
            if state.dim() == 3 and state.size(1) == 1:
                state = state.squeeze(1)
            output = self.policy_head(state)
        else:
            output = self.forward(inputs)
        action_probs = output['action_probs']  # [B, 5]
        values = output['value'].squeeze(-1)    # [B]

        # 选中动作的log_prob
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)

        # 熵（鼓励探索）
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)

        return {
            'log_probs': log_probs,
            'values': values,
            'entropy': entropy,
        }

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """保存模型checkpoint"""
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.config,
            'metadata': metadata or {},
        }
        torch.save(checkpoint, path)
        print(f"[PolicyNetwork] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, strict: bool = True):
        """加载模型checkpoint，兼容PolicyNetwork和PPOTrainer保存格式。"""
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state'))
        if state_dict is None:
            raise KeyError("Checkpoint must contain 'state_dict' or 'model_state'.")
        self.load_state_dict(state_dict, strict=strict)
        print(f"[PolicyNetwork] Checkpoint loaded from {path}")
        metadata = checkpoint.get('metadata', {})
        if not metadata:
            metadata = {
                key: checkpoint[key]
                for key in ['step_count', 'update_count', 'episode_count', 'best_reward']
                if key in checkpoint
            }
        return metadata
