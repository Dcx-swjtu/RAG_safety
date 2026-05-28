"""
Reward Function（多目标奖励函数）

设计:
R = w1 * R_correctness + w2 * R_safety + w3 * R_efficiency + w4 * R_verification

支持自适应权重调度:
- 前期: 安全优先 (safety权重高)
- 后期: 效率优先 (efficiency权重高)

奖励信号:
+10  正确检测攻击
+5   正确通过良性
-10  漏检攻击
-8   误杀良性
-cost 验证成本
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RewardComponents:
    """奖励分量明细"""
    correctness: float = 0.0
    safety: float = 0.0
    efficiency: float = 0.0
    verification: float = 0.0
    total: float = 0.0


@dataclass
class StepInfo:
    """步骤信息（用于奖励计算）"""
    is_attacked: bool = False
    attack_detected: bool = False
    attack_succeeded: bool = False
    answer_correct: bool = False
    verification_cost_ms: float = 0.0
    action_taken: int = 0
    false_positive: bool = False
    false_negative: bool = False
    final_step: bool = False
    ground_truth: str = ""
    generated_answer: str = ""
    target_answer: str = ""


class RewardFunction:
    """
    多目标奖励函数

    四个奖励分量:
    1. R_correctness: 答案正确性
    2. R_safety: 安全性（检测攻击能力）
    3. R_efficiency: 效率（验证成本控制）
    4. R_verification: 验证准确性
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化奖励函数

        Args:
            config: 配置字典
                - weights: Dict[str, float] 各分量权重
                - adaptive_schedule: bool 是否启用自适应权重
                - cost_per_verification: float 每次验证成本
                - max_steps: int 最大步数
        """
        self.config = config or {}

        # 权重配置
        default_weights = {
            'correctness': 0.5,
            'safety': 0.3,
            'efficiency': 0.15,
            'verification': 0.05,
        }
        self.weights = self.config.get('weights', default_weights)
        self.adaptive_schedule = self.config.get('adaptive_schedule', True)
        self.cost_per_verification = self.config.get('cost_per_verification', 1.0)
        self.max_steps = self.config.get('max_steps', 20)

        # 跟踪训练进度（用于自适应权重）
        self.global_step = 0
        self.metrics_history = []

    def compute(
        self,
        step_info: StepInfo,
        global_step: Optional[int] = None,
    ) -> RewardComponents:
        """
        计算多目标奖励

        Args:
            step_info: 步骤信息
            global_step: 全局步数（用于自适应权重）

        Returns:
            RewardComponents包含各分量和总奖励
        """
        if global_step is not None:
            self.global_step = global_step

        # 计算各分量奖励
        r_correctness = self._compute_correctness_reward(step_info)
        r_safety = self._compute_safety_reward(step_info)
        r_efficiency = self._compute_efficiency_reward(step_info)
        r_verification = self._compute_verification_reward(step_info)

        # 自适应权重
        if self.adaptive_schedule and global_step is not None:
            weights = self._adaptive_weights(global_step)
        else:
            weights = self.weights

        # 加权求和
        total_reward = (
            weights['correctness'] * r_correctness +
            weights['safety'] * r_safety +
            weights['efficiency'] * r_efficiency +
            weights['verification'] * r_verification
        )

        # 终止奖励/惩罚。攻击样本上，成功过滤/拒答且未被目标答案劫持
        # 也算任务成功，否则 PPO 会错误惩罚安全动作。
        if step_info.final_step:
            defense_success = (
                step_info.is_attacked
                and step_info.attack_detected
                and not step_info.attack_succeeded
            )
            if step_info.answer_correct or defense_success:
                total_reward += 10.0
            else:
                total_reward -= 10.0

            # 攻击检测奖励
            if step_info.is_attacked and step_info.attack_detected:
                total_reward += 5.0

        return RewardComponents(
            correctness=r_correctness,
            safety=r_safety,
            efficiency=r_efficiency,
            verification=r_verification,
            total=total_reward,
        )

    def _compute_correctness_reward(self, step_info: StepInfo) -> float:
        """
        答案正确性奖励

        奖励逻辑:
        - 答案正确: +1.0
        - 答案错误: -1.0
        - 非终止步骤: 0.0
        """
        if not step_info.final_step:
            return 0.0

        if step_info.answer_correct:
            return 1.0
        else:
            return -1.0

    def _compute_safety_reward(self, step_info: StepInfo) -> float:
        """
        安全性奖励

        奖励逻辑（核心！）:
        +10  正确检测攻击（REJECT且确实有攻击）
        +5   正确通过良性（无攻击且回答正确）
        -10  漏检攻击（有攻击但未检测）
        -8   误杀良性（REJECT但无攻击）
        """
        reward = 0.0

        if step_info.final_step:
            # 情况1: 确实有攻击
            if step_info.is_attacked:
                if step_info.attack_detected:
                    # 正确检测攻击
                    reward += 10.0
                else:
                    # 漏检攻击
                    reward -= 10.0
            else:
                # 情况2: 无攻击（良性查询）
                if step_info.false_positive:
                    # 误杀良性
                    reward -= 8.0
                elif step_info.answer_correct:
                    # 正确通过良性
                    reward += 5.0

        # 中间步骤: 执行验证动作给予小奖励
        if step_info.action_taken in [1, 2, 3]:  # DROP/RERANK document-level defenses
            if step_info.is_attacked:
                reward += 0.3  # 执行验证的安全奖励

        return reward

    def _compute_efficiency_reward(self, step_info: StepInfo) -> float:
        """
        效率奖励

        惩罚过度验证:
        - 每次验证动作成本惩罚
        - REJECT/SKIP的成本较低
        - LIGHT/DEEP/EXPAND成本递增
        """
        action_costs = {
            0: 0.0,    # KEEP_DOCS: no extra verification cost
            1: 1.0,    # DROP_SUSPECT_DOCS: low cost
            2: 2.0,    # RERANK_DOCS: low/medium cost
            3: 3.0,    # DROP_AND_RERANK: medium cost
            4: 0.5,    # ABSTAIN: low compute cost, utility handled elsewhere
        }

        cost = action_costs.get(step_info.action_taken, 1.0)

        # 良性查询但做了不必要的验证 -> 额外惩罚
        if not step_info.is_attacked and step_info.action_taken in [2, 3]:
            cost *= 1.5

        return -cost * 0.01  # 缩放成本惩罚

    def _compute_verification_reward(self, step_info: StepInfo) -> float:
        """
        验证准确性奖励

        奖励:
        - 验证动作与实际情况匹配
        - 高风险查询选择深度验证
        - 低风险查询选择跳过/轻量
        """
        reward = 0.0

        if not step_info.final_step:
            return reward

        # 理想的动作映射
        ideal_action = self._ideal_action(step_info)

        if step_info.action_taken == ideal_action:
            reward += 1.0
        elif abs(step_info.action_taken - ideal_action) == 1:
            reward += 0.5  # 接近理想

        return reward

    def _ideal_action(self, step_info: StepInfo) -> int:
        """
        计算理想动作（用于验证准确性奖励）

        理想策略:
        - 无攻击 + 低风险: SKIP (0)
        - 无攻击 + 中风险: LIGHT (1)
        - 有攻击 + 可检测: DEEP (2) 或 REJECT (4)
        - 不确定: EXPAND (3)
        """
        if step_info.is_attacked:
            if step_info.attack_detected:
                return 4  # REJECT
            else:
                return 2  # DEEP（尝试检测）
        else:
            return 0  # SKIP（良性查询）

    def _adaptive_weights(self, global_step: int) -> Dict[str, float]:
        """
        自适应权重调度

        策略:
        - 前期(0-30K步): 安全优先
        - 中期(30K-60K步): 平衡
        - 后期(60K+步): 效率优先
        """
        if global_step < 30000:
            # 安全优先
            return {
                'correctness': 0.4,
                'safety': 0.4,
                'efficiency': 0.1,
                'verification': 0.1,
            }
        elif global_step < 60000:
            # 平衡
            return {
                'correctness': 0.45,
                'safety': 0.3,
                'efficiency': 0.15,
                'verification': 0.1,
            }
        else:
            # 效率优先
            return {
                'correctness': 0.5,
                'safety': 0.2,
                'efficiency': 0.2,
                'verification': 0.1,
            }

    def update_metrics(self, metrics: Dict[str, float]):
        """更新指标历史（用于自适应权重）"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def get_weight_schedule(self, total_steps: int) -> List[Dict[str, float]]:
        """
        获取完整权重调度计划

        Returns:
            每个step的权重字典列表
        """
        schedule = []
        for step in range(total_steps):
            schedule.append(self._adaptive_weights(step))
        return schedule
