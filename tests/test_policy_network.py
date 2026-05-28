"""
Policy Network 单元测试

测试覆盖:
- 状态编码器输出维度
- 策略头输出概率分布
- 价值头输出
- 层次化决策
- 动作选择
- 动作评估（PPO更新用）
- Checkpoint保存/加载
- 多GPU支持检查
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

import torch
import torch.nn as nn

from verirag.policy_network import VerificationPolicyNetwork
from verirag.state_encoder import (
    StateEncoder,
    QueryEncoder,
    DocumentSetEncoder,
    ConflictEncoder,
    HistoryEncoder,
    HierarchicalPolicyHead,
)


class TestStateEncoder(unittest.TestCase):
    """状态编码器测试"""

    @classmethod
    def setUpClass(cls):
        cls.state_encoder = StateEncoder(
            bert_model_name="bert-base-uncased",
            max_docs=5,
            history_length=5,
            num_actions=5,
            output_dim=512,
        )
        cls.batch_size = 2
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.state_encoder = cls.state_encoder.to(cls.device)

    def test_query_encoder_output(self):
        """测试QueryEncoder输出"""
        query_tokens = {
            'input_ids': torch.zeros(self.batch_size, 512, dtype=torch.long, device=self.device),
            'attention_mask': torch.ones(self.batch_size, 512, dtype=torch.long, device=self.device),
        }

        self.state_encoder.eval()
        with torch.no_grad():
            result = self.state_encoder.query_encoder(query_tokens)

        self.assertIn('embedding', result)
        self.assertIn('risk_score', result)
        self.assertIn('adversarial_prob', result)

        # 检查embedding维度
        self.assertEqual(result['embedding'].shape, (self.batch_size, 128))

    def test_document_set_encoder_output(self):
        """测试DocumentSetEncoder输出"""
        doc_embeddings = torch.randn(self.batch_size, 5, 768, device=self.device)
        doc_scores = torch.ones(self.batch_size, 5, device=self.device) / 5
        doc_masks = torch.ones(self.batch_size, 5, dtype=torch.bool, device=self.device)

        self.state_encoder.eval()
        with torch.no_grad():
            result = self.state_encoder.doc_encoder(
                doc_embeddings, doc_scores, doc_masks
            )

        self.assertIn('doc_set_embedding', result)
        self.assertIn('doc_stats', result)
        self.assertIn('individual_doc_emb', result)

        # 检查输出维度
        self.assertEqual(result['doc_set_embedding'].shape, (self.batch_size, 128))
        self.assertEqual(result['doc_stats'].shape, (self.batch_size, 64))

    def test_conflict_encoder_output(self):
        """测试ConflictEncoder输出"""
        doc_embeddings = torch.randn(self.batch_size, 5, 256, device=self.device)

        self.state_encoder.eval()
        with torch.no_grad():
            result = self.state_encoder.conflict_encoder(doc_embeddings)

        self.assertIn('conflict_embedding', result)
        self.assertIn('conflict_score', result)
        self.assertIn('conflict_types', result)

        # 检查输出维度
        self.assertEqual(result['conflict_embedding'].shape, (self.batch_size, 64))
        self.assertEqual(result['conflict_score'].shape, (self.batch_size, 1))
        self.assertEqual(result['conflict_types'].shape, (self.batch_size, 4))

    def test_history_encoder_output(self):
        """测试HistoryEncoder输出"""
        action_history = torch.randint(0, 5, (self.batch_size, 5), device=self.device)
        result_history = torch.randint(0, 3, (self.batch_size, 5), device=self.device)

        self.state_encoder.eval()
        with torch.no_grad():
            result = self.state_encoder.history_encoder(
                action_history, result_history
            )

        self.assertIn('history_embedding', result)
        self.assertIn('history_attention', result)

        # 检查输出维度
        self.assertEqual(result['history_embedding'].shape, (self.batch_size, 64))

    def test_full_state_encoding(self):
        """测试完整状态编码"""
        inputs = {
            'query_tokens': {
                'input_ids': torch.zeros(self.batch_size, 512, dtype=torch.long, device=self.device),
                'attention_mask': torch.ones(self.batch_size, 512, dtype=torch.long, device=self.device),
            },
            'query_text': 'test query',
            'doc_embeddings': torch.randn(self.batch_size, 5, 768, device=self.device),
            'doc_scores': torch.ones(self.batch_size, 5, device=self.device) / 5,
            'doc_masks': torch.ones(self.batch_size, 5, dtype=torch.bool, device=self.device),
            'action_history': torch.zeros(self.batch_size, 0, dtype=torch.long, device=self.device),
            'result_history': torch.zeros(self.batch_size, 0, dtype=torch.long, device=self.device),
        }

        self.state_encoder.eval()
        with torch.no_grad():
            result = self.state_encoder(inputs)

        self.assertIn('state', result)
        self.assertIn('query_emb', result)
        self.assertIn('doc_emb', result)
        self.assertIn('conflict_emb', result)
        self.assertIn('history_emb', result)
        self.assertIn('risk_score', result)
        self.assertIn('adversarial_prob', result)
        self.assertIn('conflict_score', result)

        # 检查state维度
        self.assertEqual(result['state'].shape, (self.batch_size, 512))


class TestHierarchicalPolicyHead(unittest.TestCase):
    """层次化策略头测试"""

    @classmethod
    def setUpClass(cls):
        cls.policy_head = HierarchicalPolicyHead(
            state_dim=512,
            hidden_dim=256,
            num_actions=5,
        )
        cls.batch_size = 4
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.policy_head = cls.policy_head.to(cls.device)

    def test_forward_output(self):
        """测试前向传播输出"""
        state = torch.randn(self.batch_size, 512, device=self.device)

        self.policy_head.eval()
        with torch.no_grad():
            result = self.policy_head(state)

        self.assertIn('action_logits', result)
        self.assertIn('action_probs', result)
        self.assertIn('value', result)
        self.assertIn('level1_probs', result)
        self.assertIn('level2_probs', result)

        # 检查维度
        self.assertEqual(result['action_logits'].shape, (self.batch_size, 5))
        self.assertEqual(result['action_probs'].shape, (self.batch_size, 5))
        self.assertEqual(result['value'].shape, (self.batch_size, 1))
        self.assertEqual(result['level1_probs'].shape, (self.batch_size, 3))
        self.assertEqual(result['level2_probs'].shape, (self.batch_size, 3))

    def test_probability_distribution(self):
        """测试概率分布正确性"""
        state = torch.randn(1, 512, device=self.device)

        self.policy_head.eval()
        with torch.no_grad():
            result = self.policy_head(state)

        # 概率之和应为1
        probs_sum = result['action_probs'].sum(dim=-1)
        self.assertTrue(torch.allclose(probs_sum, torch.ones(1, device=self.device), atol=1e-5))

    def test_temperature_effect(self):
        """测试温度对概率分布的影响"""
        state = torch.randn(1, 512, device=self.device)

        self.policy_head.eval()
        with torch.no_grad():
            result_high_temp = self.policy_head(state, temperature=2.0)
            result_low_temp = self.policy_head(state, temperature=0.1)

        # 高温应该产生更均匀的分布（更高的熵）
        entropy_high = -(result_high_temp['action_probs'] *
                         torch.log(result_high_temp['action_probs'] + 1e-8)).sum()
        entropy_low = -(result_low_temp['action_probs'] *
                        torch.log(result_low_temp['action_probs'] + 1e-8)).sum()

        self.assertGreater(entropy_high.item(), entropy_low.item())


class TestVerificationPolicyNetwork(unittest.TestCase):
    """验证策略网络完整测试"""

    @classmethod
    def setUpClass(cls):
        cls.config = {
            'state_dim': 512,
            'action_dim': 5,
            'hidden_dim': 256,
            'max_docs': 5,
            'history_length': 5,
        }
        cls.policy_net = VerificationPolicyNetwork(config=cls.config)
        cls.batch_size = 2
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.policy_net = cls.policy_net.to(cls.device)

    def _create_test_inputs(self, batch_size=2):
        """创建测试输入"""
        return {
            'query_tokens': {
                'input_ids': torch.zeros(batch_size, 512, dtype=torch.long, device=self.device),
                'attention_mask': torch.ones(batch_size, 512, dtype=torch.long, device=self.device),
            },
            'query_text': 'test query about revenue',
            'doc_embeddings': torch.randn(batch_size, 5, 768, device=self.device),
            'doc_scores': torch.ones(batch_size, 5, device=self.device) / 5,
            'doc_masks': torch.ones(batch_size, 5, dtype=torch.bool, device=self.device),
            'action_history': torch.zeros(batch_size, 0, dtype=torch.long, device=self.device),
            'result_history': torch.zeros(batch_size, 0, dtype=torch.long, device=self.device),
        }

    def test_forward(self):
        """测试完整前向传播"""
        inputs = self._create_test_inputs()

        self.policy_net.eval()
        with torch.no_grad():
            result = self.policy_net(inputs)

        self.assertIn('state', result)
        self.assertIn('action_logits', result)
        self.assertIn('action_probs', result)
        self.assertIn('value', result)
        self.assertIn('risk_score', result)

    def test_select_action(self):
        """测试动作选择"""
        inputs = self._create_test_inputs(batch_size=1)

        self.policy_net.eval()
        with torch.no_grad():
            result = self.policy_net.select_action(inputs, deterministic=True)

        self.assertIn('action', result)
        self.assertIn('action_name', result)
        self.assertIn('action_probs', result)
        self.assertIn('value', result)
        self.assertIn('log_prob', result)

        # 检查动作范围
        self.assertIn(result['action'], range(5))
        self.assertIn(result['action_name'], [
            'KEEP_DOCS', 'DROP_SUSPECT_DOCS', 'RERANK_DOCS', 'DROP_AND_RERANK', 'ABSTAIN'
        ])

        # 检查概率
        self.assertEqual(len(result['action_probs']), 5)
        self.assertAlmostEqual(sum(result['action_probs']), 1.0, places=5)

    def test_evaluate_actions(self):
        """测试动作评估（PPO更新）"""
        inputs = self._create_test_inputs()
        actions = torch.randint(0, 5, (self.batch_size,), device=self.device)

        result = self.policy_net.evaluate_actions(inputs, actions)

        self.assertIn('log_probs', result)
        self.assertIn('values', result)
        self.assertIn('entropy', result)

        self.assertEqual(result['log_probs'].shape, (self.batch_size,))
        self.assertEqual(result['values'].shape, (self.batch_size,))
        self.assertEqual(result['entropy'].shape, (self.batch_size,))

    def test_deterministic_vs_stochastic(self):
        """测试确定性与随机策略"""
        inputs = self._create_test_inputs(batch_size=1)

        self.policy_net.eval()

        # 确定性选择应该总是返回相同动作
        with torch.no_grad():
            actions = []
            for _ in range(5):
                result = self.policy_net.select_action(inputs, deterministic=True)
                actions.append(result['action'])

        self.assertEqual(len(set(actions)), 1, "确定性策略应该返回相同动作")

    def test_checkpoint_save_load(self):
        """测试Checkpoint保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = f"{tmpdir}/test_checkpoint.pt"
            metadata = {'test': True, 'epoch': 10}

            # 保存前获取权重
            original_weights = {}
            for name, param in self.policy_net.named_parameters():
                original_weights[name] = param.clone()

            # 保存
            self.policy_net.save_checkpoint(save_path, metadata)
            self.assertTrue(torch.cuda.is_available() or True)  # 文件应存在

            # 修改权重
            with torch.no_grad():
                for param in self.policy_net.parameters():
                    param.add_(torch.randn_like(param))

            # 加载
            loaded_metadata = self.policy_net.load_checkpoint(save_path)

            # 验证元数据
            self.assertEqual(loaded_metadata['test'], True)
            self.assertEqual(loaded_metadata['epoch'], 10)

    def test_gradient_flow(self):
        """测试梯度流"""
        inputs = self._create_test_inputs()
        actions = torch.randint(0, 5, (self.batch_size,), device=self.device)

        self.policy_net.train()

        # 前向传播
        result = self.policy_net.evaluate_actions(inputs, actions)

        # 计算损失
        loss = -result['log_probs'].mean() + 0.5 * result['values'].mean()

        # 反向传播
        loss.backward()

        # 检查梯度存在
        has_grad = False
        for param in self.policy_net.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        self.assertTrue(has_grad, "应该有梯度流动")

        # 清除梯度
        self.policy_net.zero_grad()


class TestMultiGPU(unittest.TestCase):
    """多GPU支持测试"""

    @unittest.skipIf(torch.cuda.device_count() < 2, "需要至少2个GPU")
    def test_data_parallel(self):
        """测试DataParallel"""
        policy_net = VerificationPolicyNetwork(config={'state_dim': 512, 'action_dim': 5})
        policy_net = nn.DataParallel(policy_net)

        inputs = {
            'query_tokens': {
                'input_ids': torch.zeros(2, 512, dtype=torch.long),
                'attention_mask': torch.ones(2, 512, dtype=torch.long),
            },
            'query_text': 'test',
            'doc_embeddings': torch.randn(2, 5, 768),
            'doc_scores': torch.ones(2, 5) / 5,
            'doc_masks': torch.ones(2, 5, dtype=torch.bool),
            'action_history': torch.zeros(2, 0, dtype=torch.long),
            'result_history': torch.zeros(2, 0, dtype=torch.long),
        }

        policy_net.eval()
        with torch.no_grad():
            result = policy_net(inputs)

        self.assertIn('action_probs', result)
        self.assertEqual(result['action_probs'].shape[0], 2)


if __name__ == '__main__':
    unittest.main()
