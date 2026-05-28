"""
PPO Trainer（PPO训练器）

支持:
- GAE (Generalized Advantage Estimation) advantage计算
- Value clipping
- Entropy bonus（鼓励探索）
- Learning rate scheduling
- Multi-GPU训练 (DataParallel/DistributedDataParallel)
- Checkpoint保存/恢复

PPO算法流程:
1. Collect Rollout: 与环境交互收集 (s, a, r, v, log_p) 序列
2. Compute GAE: 计算优势函数
3. Update Policy: 多次epoch更新策略
4. Update Value: 更新价值函数
5. Repeat
"""

import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .policy_network import VerificationPolicyNetwork
from .environment import RAGDefenseEnv


# ==================== Rollout Buffer ====================

@dataclass
class Transition:
    """单个转移"""
    state: Dict[str, torch.Tensor]  # 状态（存储state张量，不是完整输入字典）
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool


class RolloutBuffer:
    """
    Rollout数据缓冲区

    存储一个rollout的 (s, a, r, v, log_p, done) 序列
    支持GAE计算和批量采样
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.clear()

    def clear(self):
        """清空缓冲区"""
        self.states = []          # state张量列表 [B, state_dim]
        self.actions = []         # 动作索引 [B]
        self.rewards = []         # 奖励 [B]
        self.values = []          # 价值估计 [B]
        self.log_probs = []       # log概率 [B]
        self.dones = []           # 终止标志 [B]
        self.size = 0

    def store(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """存储一个转移"""
        if self.size >= self.capacity:
            return  # 缓冲区满

        self.states.append(state.detach().cpu())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.size += 1

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算returns和advantages（GAE）

        Args:
            gamma: 折扣因子
            gae_lambda: GAE lambda
            last_value: 最后一个状态的V值（用于bootstrap）

        Returns:
            (returns, advantages) 张量
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        # 反向计算GAE
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + gamma * next_value - values[t]

            # GAE
            if dones[t]:
                last_gae = delta
            else:
                last_gae = delta + gamma * gae_lambda * last_gae

            advantages[t] = last_gae

        # Returns = advantages + values
        returns = advantages + values[:-1]

        return torch.FloatTensor(returns), torch.FloatTensor(advantages)

    def get_batches(
        self,
        batch_size: int,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        num_epochs: int = 1,
    ):
        """
        生成训练批次

        Args:
            batch_size: 批次大小
            returns: return值
            advantages: advantage值
            num_epochs: epoch数量

        Yields:
            (state_batch, action_batch, return_batch, advantage_batch, old_log_prob_batch)
        """
        n = self.size
        indices = np.arange(n)

        for epoch in range(num_epochs):
            # 每个epoch打乱
            np.random.shuffle(indices)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]

                state_batch = torch.stack([self.states[i] for i in batch_idx])
                action_batch = torch.LongTensor([self.actions[i] for i in batch_idx])
                return_batch = returns[batch_idx]
                advantage_batch = advantages[batch_idx]
                old_log_prob_batch = torch.FloatTensor([self.log_probs[i] for i in batch_idx])

                yield state_batch, action_batch, return_batch, advantage_batch, old_log_prob_batch


# ==================== PPO Trainer ====================

class PPOTrainer:
    """
    PPO训练器

    完整训练流程:
    1. 初始化Policy Network和Optimizer
    2. 循环:
        a. 收集rollout数据
        b. 计算GAE
        c. 多次epoch更新policy和value
        d. 学习率调整
        e. 保存checkpoint
    3. 保存最终模型
    """

    def __init__(
        self,
        policy_network: VerificationPolicyNetwork,
        env: RAGDefenseEnv,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化PPO训练器

        Args:
            policy_network: 策略网络
            env: RL环境
            config: 训练配置
                - lr: float (默认3e-4)
                - gamma: float (默认0.99)
                - gae_lambda: float (默认0.95)
                - clip_epsilon: float (默认0.2)
                - entropy_coef: float (默认0.01)
                - value_coef: float (默认0.5)
                - max_grad_norm: float (默认0.5)
                - batch_size: int (默认64)
                - rollout_length: int (默认200)
                - epochs_per_update: int (默认4)
                - total_steps: int (默认100000)
                - lr_schedule: str (默认"cosine")
                - use_multi_gpu: bool (默认True)
                - checkpoint_dir: str (默认"./checkpoints")
                - checkpoint_freq: int (默认1000)
                - log_freq: int (默认100)
        """
        self.config = config or {}
        self.env = env

        # 训练参数。PyYAML may parse values like 3e-4 as strings,
        # so cast explicitly before handing them to torch.
        def as_float(name: str, default: float) -> float:
            return float(self.config.get(name, default))

        def as_int(name: str, default: int) -> int:
            return int(self.config.get(name, default))

        self.lr = as_float('lr', 3e-4)
        self.gamma = as_float('gamma', 0.99)
        self.gae_lambda = as_float('gae_lambda', 0.95)
        self.clip_epsilon = as_float('clip_epsilon', 0.2)
        self.entropy_coef = as_float('entropy_coef', 0.01)
        self.value_coef = as_float('value_coef', 0.5)
        self.max_grad_norm = as_float('max_grad_norm', 0.5)
        self.batch_size = as_int('batch_size', 64)
        self.rollout_length = as_int('rollout_length', 200)
        self.epochs_per_update = as_int('epochs_per_update', 4)
        self.total_steps = as_int('total_steps', 100000)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 策略网络（支持多GPU）
        self.policy_network = policy_network.to(self.device)
        self.use_multi_gpu = self.config.get('use_multi_gpu', True) and torch.cuda.device_count() > 1

        if self.use_multi_gpu:
            print(f"[PPOTrainer] Using {torch.cuda.device_count()} GPUs")
            self.policy_network = DataParallel(self.policy_network)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.lr,
            eps=1e-5,
        )

        # 学习率调度
        lr_schedule = self.config.get('lr_schedule', 'cosine')
        if lr_schedule == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps,
                eta_min=self.lr * 0.1,
            )
        elif lr_schedule == 'linear':
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.total_steps,
            )
        else:
            self.scheduler = None

        # Rollout缓冲区
        self.rollout_buffer = RolloutBuffer(capacity=self.rollout_length)

        # Checkpoint
        self.checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_freq = self.config.get('checkpoint_freq', 1000)

        # 日志
        self.log_freq = self.config.get('log_freq', 100)
        self.step_count = 0
        self.update_count = 0
        self.episode_count = 0
        self.total_reward = 0.0

        # 训练历史
        self.loss_history = []
        self.reward_history = []
        self.value_history = []

    def collect_rollout(self, num_steps: int) -> Dict[str, Any]:
        """
        收集rollout数据

        Args:
            num_steps: 收集步数

        Returns:
            统计信息字典
        """
        self.rollout_buffer.clear()

        state = self.env.reset()
        episode_rewards = []
        episode_reward = 0.0

        for step in range(num_steps):
            self.step_count += 1

            # 将state移到GPU
            state_gpu = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state_gpu[k] = v.to(self.device)
                else:
                    state_gpu[k] = v

            # Policy选择动作
            with torch.no_grad():
                decision = self.policy_network.module.select_action(state_gpu) if self.use_multi_gpu \
                    else self.policy_network.select_action(state_gpu)

            action = decision['action']
            log_prob = decision['log_prob']
            value = decision['value']

            # 环境步进
            next_state, reward, done, info = self.env.step(action)

            # 存储transition
            # 只存储state张量（节省内存）
            state_tensor = decision['state'] if 'state' in decision else torch.zeros(1, 512)
            self.rollout_buffer.store(
                state=state_tensor,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
            )

            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                self.episode_count += 1
                state = self.env.reset()
            else:
                state = next_state

        stats = {
            'num_steps': num_steps,
            'num_episodes': len(episode_rewards),
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'max_episode_reward': np.max(episode_rewards) if episode_rewards else 0.0,
            'min_episode_reward': np.min(episode_rewards) if episode_rewards else 0.0,
        }

        return stats

    def update(self) -> Dict[str, float]:
        """
        PPO更新

        Returns:
            损失统计
        """
        if self.rollout_buffer.size == 0:
            return {}

        self.update_count += 1

        # 计算returns和advantages
        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # 归一化advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 移到GPU
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # 多epoch更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_batches = 0

        for batch in self.rollout_buffer.get_batches(
            self.batch_size, returns, advantages, self.epochs_per_update
        ):
            state_batch, action_batch, return_batch, advantage_batch, old_log_prob_batch = batch

            # 移到GPU
            state_batch = state_batch.to(self.device)
            action_batch = action_batch.to(self.device)
            return_batch = return_batch.to(self.device)
            advantage_batch = advantage_batch.to(self.device)
            old_log_prob_batch = old_log_prob_batch.to(self.device)

            # 构建policy输入
            policy_inputs = {'state': state_batch}

            # 评估当前策略下的动作
            eval_out = self.policy_network.module.evaluate_actions(policy_inputs, action_batch) if self.use_multi_gpu \
                else self.policy_network.evaluate_actions(policy_inputs, action_batch)

            new_log_probs = eval_out['log_probs']
            values = eval_out['values']
            entropy = eval_out['entropy'].mean()

            # Policy loss (PPO clipped)
            ratio = torch.exp(new_log_probs - old_log_prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = F.mse_loss(values, return_batch)

            # Total loss
            loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.use_multi_gpu:
                nn.utils.clip_grad_norm_(self.policy_network.module.parameters(), self.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # 统计
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
            num_batches += 1

        # 学习率调整
        if self.scheduler is not None:
            self.scheduler.step()

        stats = {
            'policy_loss': total_policy_loss / max(num_batches, 1),
            'value_loss': total_value_loss / max(num_batches, 1),
            'entropy': total_entropy / max(num_batches, 1),
            'total_loss': total_loss / max(num_batches, 1),
            'num_batches': num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

        self.loss_history.append(stats)
        return stats

    def train(self, total_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        完整训练循环

        Args:
            total_steps: 总训练步数（默认使用config中的值）

        Returns:
            训练统计
        """
        if total_steps is None:
            total_steps = self.total_steps

        print(f"[PPOTrainer] Starting training for {total_steps} steps")
        print(f"[PPOTrainer] Device: {self.device}")
        print(f"[PPOTrainer] Rollout length: {self.rollout_length}")
        print(f"[PPOTrainer] Batch size: {self.batch_size}")
        print(f"[PPOTrainer] Epochs per update: {self.epochs_per_update}")

        start_time = time.time()
        best_reward = float('-inf')

        while self.step_count < total_steps:
            # Step 1: 收集rollout
            rollout_stats = self.collect_rollout(self.rollout_length)

            # Step 2: PPO更新
            update_stats = self.update()

            # 记录奖励
            mean_reward = rollout_stats.get('mean_episode_reward', 0.0)
            self.reward_history.append(mean_reward)

            # 日志
            if self.step_count % self.log_freq == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.step_count / max(elapsed, 1e-6)

                print(
                    f"[Step {self.step_count}/{total_steps}] "
                    f"Reward: {mean_reward:.3f} | "
                    f"Policy Loss: {update_stats.get('policy_loss', 0):.4f} | "
                    f"Value Loss: {update_stats.get('value_loss', 0):.4f} | "
                    f"Entropy: {update_stats.get('entropy', 0):.4f} | "
                    f"LR: {update_stats.get('learning_rate', 0):.6f} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

            # Checkpoint
            if self.step_count % self.checkpoint_freq == 0:
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save_checkpoint('best_model.pt')

                self.save_checkpoint(f'checkpoint_{self.step_count}.pt')

        # 保存最终模型
        self.save_checkpoint('final_model.pt')

        elapsed = time.time() - start_time
        print(f"[PPOTrainer] Training completed in {elapsed:.1f}s")
        print(f"[PPOTrainer] Total steps: {self.step_count}")
        print(f"[PPOTrainer] Total episodes: {self.episode_count}")
        print(f"[PPOTrainer] Best reward: {best_reward:.3f}")

        return {
            'total_steps': self.step_count,
            'total_episodes': self.episode_count,
            'best_reward': best_reward,
            'final_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            'training_time': elapsed,
        }

    def _to_checkpoint_safe(self, value: Any) -> Any:
        """Convert logs/config metadata to torch weights-only safe values."""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                self._to_checkpoint_safe(key): self._to_checkpoint_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._to_checkpoint_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def save_checkpoint(self, filename: str, metadata: Optional[Dict] = None):
        """保存checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)

        # 保存模型（unwrap DataParallel）
        if self.use_multi_gpu:
            model_state = self.policy_network.module.state_dict()
        else:
            model_state = self.policy_network.state_dict()

        checkpoint = {
            'step': int(self.step_count),
            'step_count': int(self.step_count),
            'update_count': int(self.update_count),
            'episode_count': int(self.episode_count),
            'optimizer_state': self._to_checkpoint_safe(self.optimizer.state_dict()),
            'scheduler_state': self._to_checkpoint_safe(self.scheduler.state_dict()) if self.scheduler else None,
            'reward_history': self._to_checkpoint_safe(self.reward_history),
            'loss_history': self._to_checkpoint_safe(self.loss_history),
            'config': self._to_checkpoint_safe(self.config),
            'metadata': self._to_checkpoint_safe(metadata or {}),
            'model_state': model_state,
            'state_dict': model_state,
        }

        torch.save(checkpoint, path)
        print(f"[PPOTrainer] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """加载checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        # 加载模型
        if self.use_multi_gpu:
            self.policy_network.module.load_state_dict(checkpoint['model_state'])
        else:
            self.policy_network.load_state_dict(checkpoint['model_state'])

        # 加载optimizer
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # 加载scheduler
        if 'scheduler_state' in checkpoint and self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        # 加载训练状态
        self.step_count = checkpoint.get('step', 0)
        self.update_count = checkpoint.get('update_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.reward_history = checkpoint.get('reward_history', [])
        self.loss_history = checkpoint.get('loss_history', [])

        print(f"[PPOTrainer] Checkpoint loaded: {path} (step={self.step_count})")
        return checkpoint.get('metadata', {})

    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        return {
            'step_count': self.step_count,
            'update_count': self.update_count,
            'episode_count': self.episode_count,
            'mean_reward_last_100': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            'mean_policy_loss_last_100': np.mean([l.get('policy_loss', 0) for l in self.loss_history[-100:]]) if self.loss_history else 0.0,
            'mean_value_loss_last_100': np.mean([l.get('value_loss', 0) for l in self.loss_history[-100:]]) if self.loss_history else 0.0,
        }


# 导入F
import torch.nn.functional as F
