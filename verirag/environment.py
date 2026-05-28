"""
RL Environment（RAG防御环境）

State: [query_risk, doc_stats, conflict_indicators, history]
Action: {SKIP, LIGHT, DEEP, EXPAND, REJECT}
Reward: 多目标奖励

环境组成:
- 查询生成器（从数据集采样查询）
- 攻击模拟器（参数化注入攻击）
- RAG检索器（模拟文档检索）
- 答案生成器（模拟答案生成）
- 奖励计算器（多目标奖励函数）
"""

import random
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from .reward_function import RewardFunction, RewardComponents, StepInfo
from .attack_simulator import AttackSimulator, AttackType
from .text_features import TextFeatureExtractor


class RAGDefenseEnv:
    """
    RL环境: RAG系统 + 攻击者 + 防御系统

    遵循OpenAI Gym接口:
    - reset() -> state
    - step(action) -> (next_state, reward, done, info)
    """

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        attack_simulator: AttackSimulator,
        retriever: Any,  # 检索器
        generator: Any,  # 生成器
        reward_function: RewardFunction,
        state_dim: int = 512,
        action_dim: int = 5,
        max_steps_per_episode: int = 20,
        attack_probability: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化环境

        Args:
            corpus: 文档语料库
            attack_simulator: 攻击模拟器
            retriever: 文档检索器
            generator: 答案生成器
            reward_function: 奖励函数
            state_dim: 状态维度
            action_dim: 动作维度
            max_steps_per_episode: 每episode最大步数
            attack_probability: 攻击概率
            config: 额外配置
        """
        self.corpus = corpus
        self.attack_simulator = attack_simulator
        self.retriever = retriever
        self.generator = generator
        self.reward_function = reward_function
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps_per_episode
        self.attack_probability = attack_probability
        self.config = config or {}

        # 当前episode状态
        self.current_query = None
        self.current_docs = None
        self.is_attacked = False
        self.target_answer = None
        self.ground_truth = None
        self.step_count = 0
        self.history_actions = []
        self.history_results = []

        # 真实、确定性的文本特征，替代旧版随机 doc embedding。
        feature_config = dict(self.config.get('feature_extractor', {}))
        feature_config.setdefault('max_docs', self.config.get('max_docs', 10))
        self.text_features = TextFeatureExtractor(feature_config)

        # 统计
        self.total_episodes = 0
        self.attack_success_count = 0
        self.correct_answer_count = 0

    def reset(self) -> Dict[str, torch.Tensor]:
        """
        重置环境，开始新episode

        Returns:
            初始state字典
        """
        self.step_count = 0
        self.history_actions = []
        self.history_results = []
        self.total_episodes += 1

        # 随机采样查询
        sample = random.choice(self.corpus)
        self.current_query = sample['query']
        self.ground_truth = sample.get('answer', '')

        # 检索文档
        retrieved_docs = self._retrieve_documents(self.current_query)

        # 决定是否注入攻击
        self.is_attacked = random.random() < self.attack_probability

        if self.is_attacked:
            # 选择目标错误答案
            self.target_answer = sample.get('target_answer', '')
            if not self.target_answer:
                self.target_answer = self._generate_target_answer(self.ground_truth)

            # 生成攻击文档
            attack_docs = self.attack_simulator.generate(
                query=self.current_query,
                target_answer=self.target_answer,
                attack_type=random.choice([
                    AttackType.POISONEDRAG,
                    AttackType.ONESHOT,
                    AttackType.REFINERAG,
                    AttackType.SEMANTIC_CHAMELEON,
                ]),
                original_docs=retrieved_docs,
            )

            # 注入攻击文档
            self.current_docs = self._inject_attacks(retrieved_docs, attack_docs)
        else:
            self.target_answer = None
            self.current_docs = retrieved_docs

        # 构建初始state
        state = self._build_state()

        return state

    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        执行动作

        Args:
            action: 动作索引 (0-4)
                0: SKIP - 跳过验证，直接生成
                1: LIGHT - 轻量验证
                2: DEEP - 深度验证
                3: EXPAND - 扩展检索+深度验证
                4: REJECT - 拒绝回答

        Returns:
            (next_state, reward, done, info)
        """
        self.step_count += 1
        self.history_actions.append(action)

        # 执行动作
        filtered_docs = self._execute_action(action)

        # 生成答案
        if action == 4:  # REJECT
            generated_answer = "I cannot answer this question due to security concerns."
        else:
            generated_answer = self._generate_answer(self.current_query, filtered_docs)

        # 评估结果
        answer_correct = self._evaluate_answer(generated_answer, self.ground_truth)
        attack_detected = (action == 4) or (action in [2, 3] and self.is_attacked)
        attack_succeeded = self.is_attacked and answer_correct and generated_answer == self.target_answer

        # 记录历史结果
        result = 1 if (answer_correct or attack_detected) else 2 if attack_succeeded else 0
        self.history_results.append(result)

        # 计算验证成本
        verification_cost = self._compute_verification_cost(action)

        # 构建StepInfo
        step_info = StepInfo(
            is_attacked=self.is_attacked,
            attack_detected=attack_detected,
            attack_succeeded=attack_succeeded,
            answer_correct=answer_correct,
            verification_cost_ms=verification_cost,
            action_taken=action,
            false_positive=(action == 4 and not self.is_attacked),
            false_negative=(self.is_attacked and not attack_detected),
            final_step=True,  # 单步episode
            ground_truth=self.ground_truth,
            generated_answer=generated_answer,
            target_answer=self.target_answer or "",
        )

        # 计算奖励
        reward_components = self.reward_function.compute(
            step_info=step_info,
            global_step=self.total_episodes,
        )
        reward = reward_components.total

        # 更新统计
        if attack_succeeded:
            self.attack_success_count += 1
        if answer_correct:
            self.correct_answer_count += 1

        # 构建info
        info = {
            'is_attacked': self.is_attacked,
            'attack_detected': attack_detected,
            'attack_succeeded': attack_succeeded,
            'answer_correct': answer_correct,
            'generated_answer': generated_answer,
            'ground_truth': self.ground_truth,
            'target_answer': self.target_answer,
            'action': action,
            'reward_components': {
                'correctness': reward_components.correctness,
                'safety': reward_components.safety,
                'efficiency': reward_components.efficiency,
                'verification': reward_components.verification,
            },
            'step_count': self.step_count,
        }

        # 判断终止
        done = True  # 单步episode

        # 构建next_state
        next_state = self._build_state()

        return next_state, reward, done, info

    def step_batch(
        self,
        actions: List[int],
        states: List[Dict[str, torch.Tensor]],
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[float], List[bool], List[Dict]]:
        """
        批量执行动作（用于并行环境）

        Args:
            actions: 动作列表
            states: 状态列表

        Returns:
            (next_states, rewards, dones, infos)
        """
        next_states = []
        rewards = []
        dones = []
        infos = []

        # 简化为串行执行（实际可并行化）
        for action in actions:
            ns, r, d, i = self.step(action)
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
            infos.append(i)
            # 每个action对应一个独立的episode
            if not d:
                self.reset()

        return next_states, rewards, dones, infos

    def _build_state(self) -> Dict[str, torch.Tensor]:
        """
        构建状态向量

        Returns:
            包含所有state组件的字典（供Policy Network使用）
        """
        B = 1  # batch_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Query/doc 特征：确定性文本编码，不再使用随机 embedding。
        query_tokens = self._query_tokens(self.current_query or '', device)

        docs = (self.current_docs or [])[:self.config.get('max_docs', 10)]
        n_docs = len(docs)
        doc_emb_np = self.text_features.doc_embeddings(docs)
        doc_scores_np = self.text_features.query_doc_scores(self.current_query or '', docs, doc_emb_np)
        doc_embeddings = torch.zeros(B, n_docs, 768, dtype=torch.float32, device=device)
        if n_docs > 0:
            emb = torch.from_numpy(doc_emb_np).to(device=device, dtype=torch.float32)
            if emb.shape[1] >= 768:
                doc_embeddings[0] = emb[:, :768]
            else:
                doc_embeddings[0, :, :emb.shape[1]] = emb
        doc_scores = torch.from_numpy(doc_scores_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        doc_masks = torch.ones(B, n_docs, dtype=torch.bool, device=device)

        # History
        if self.history_actions:
            K = min(len(self.history_actions), 10)
            action_history = torch.tensor(
                self.history_actions[-K:], dtype=torch.long, device=device
            ).unsqueeze(0)
            result_history = torch.tensor(
                self.history_results[-K:], dtype=torch.long, device=device
            ).unsqueeze(0)
            history_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        else:
            action_history = torch.zeros(B, 0, dtype=torch.long, device=device)
            result_history = torch.zeros(B, 0, dtype=torch.long, device=device)
            history_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)

        return {
            'query_tokens': query_tokens,
            'query_text': self.current_query,
            'doc_embeddings': doc_embeddings,
            'doc_scores': doc_scores,
            'doc_masks': doc_masks,
            'action_history': action_history,
            'result_history': result_history,
            'history_mask': history_mask,
        }


    @staticmethod
    def _query_tokens(query: str, device: str) -> Dict[str, torch.Tensor]:
        max_length = 512
        ids = []
        for token in re.findall(r"[A-Za-z0-9]+", query.lower())[:max_length]:
            value = int.from_bytes(token.encode('utf-8')[:8].ljust(8, b'0'), 'little')
            ids.append(value % 30521 + 1)
        ids = ids + [0] * (max_length - len(ids))
        mask = [1 if token_id else 0 for token_id in ids]
        return {
            'input_ids': torch.tensor([ids], dtype=torch.long, device=device),
            'attention_mask': torch.tensor([mask], dtype=torch.long, device=device),
        }

    def _execute_action(self, action: int) -> List[str]:
        """
        执行防御动作

        Args:
            action: 动作索引

        Returns:
            过滤后的文档列表
        """
        docs = self.current_docs or []

        if action == 0:  # SKIP - 直接生成，不过滤
            return docs

        elif action == 1:  # LIGHT - 轻量验证（简单规则检查）
            return self._light_verification(docs)

        elif action == 2:  # DEEP - 深度验证（交叉验证）
            return self._deep_verification(docs)

        elif action == 3:  # EXPAND - 扩展检索
            expanded_docs = self._expand_retrieval()
            return self._deep_verification(expanded_docs)

        elif action == 4:  # REJECT - 拒绝回答
            return []

        return docs

    def _light_verification(self, docs: List[str]) -> List[str]:
        """
        轻量验证: 快速规则检查

        检查:
        - 文档长度异常（过短可能是攻击）
        - 重复内容检测
        - 基本格式检查
        """
        if not docs:
            return docs

        filtered = []
        seen_hashes = set()

        for doc in docs:
            # 检查长度
            if len(doc) < 20:  # 过短的文档可能有问题
                continue

            # 去重
            doc_hash = hash(doc[:100])
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)

            filtered.append(doc)

        # 如果过滤掉太多，保留原始文档
        if len(filtered) < max(len(docs) // 2, 1):
            return docs

        return filtered

    def _deep_verification(self, docs: List[str]) -> List[str]:
        """
        深度验证: 完整交叉验证

        检查:
        - 声明一致性（数值/实体）
        - 来源可信度
        - 语义完整性
        """
        # 简化为轻量验证 + 额外检查
        filtered = self._light_verification(docs)

        # 如果检测到冲突，过滤可疑文档
        if len(filtered) >= 2:
            # 模拟冲突检测
            if self._detect_conflicts(filtered):
                # 移除最可疑的文档
                filtered = filtered[:len(filtered) // 2 + 1]

        return filtered

    def _expand_retrieval(self) -> List[str]:
        """扩展检索: 检索更多文档"""
        # 模拟扩展检索
        additional_docs = self._retrieve_documents(self.current_query, top_k=10)
        if self.current_docs:
            return self.current_docs + additional_docs
        return additional_docs

    def _detect_conflicts(self, docs: List[str]) -> bool:
        """检测文档间冲突（简化版）"""
        if len(docs) < 2:
            return False

        # 检查数值一致性
        values = []
        for doc in docs:
            nums = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', doc)
            if nums:
                values.append(nums[0])

        if len(values) >= 2 and len(set(values)) > 1:
            return True

        return False

    def _retrieve_documents(self, query: str, top_k: int = 5) -> List[str]:
        """
        检索文档（使用模拟检索器）

        Args:
            query: 查询
            top_k: 检索数量

        Returns:
            文档文本列表
        """
        if self.retriever is not None:
            try:
                return self.retriever.retrieve(query, top_k=top_k)
            except Exception:
                pass

        # 模拟检索: 从语料库随机采样
        if self.corpus:
            samples = random.sample(self.corpus, min(top_k, len(self.corpus)))
            return [s.get('text', s.get('document', '')) for s in samples]

        return []

    def _inject_attacks(self, docs: List[str], attack_docs: List[str]) -> List[str]:
        """将攻击文档注入检索结果"""
        if not attack_docs:
            return docs

        # 随机位置注入
        result = list(docs)
        for attack_doc in attack_docs:
            pos = random.randint(0, len(result))
            result.insert(pos, attack_doc)

        return result

    def _generate_answer(self, query: str, docs: List[str]) -> str:
        """
        生成答案（使用模拟生成器）
        """
        if self.generator is not None:
            try:
                return self.generator.generate(query, docs)
            except Exception:
                pass

        # 模拟生成: 随机返回答案
        if random.random() < 0.7:
            return self.ground_truth
        else:
            # 模拟生成错误答案
            wrong_answers = ["unknown", "not mentioned", "incorrect"]
            return random.choice(wrong_answers)

    def _evaluate_answer(self, generated: str, ground_truth: str) -> bool:
        """评估答案正确性"""
        if not ground_truth:
            return False
        return ground_truth.lower() in generated.lower()

    def _generate_target_answer(self, ground_truth: str) -> str:
        """生成目标错误答案"""
        # 简单替换为相反/错误答案
        wrong_answers = ["unknown", "incorrect", "false"]
        return random.choice(wrong_answers)

    def _compute_verification_cost(self, action: int) -> float:
        """计算验证成本（毫秒）"""
        costs = {
            0: 0,      # SKIP
            1: 50,     # LIGHT: ~50ms
            2: 500,    # DEEP: ~500ms
            3: 1000,   # EXPAND: ~1000ms
            4: 10,     # REJECT
        }
        return costs.get(action, 0)

    def get_statistics(self) -> Dict[str, float]:
        """获取环境统计"""
        if self.total_episodes == 0:
            return {}

        return {
            'total_episodes': self.total_episodes,
            'attack_success_rate': self.attack_success_count / max(self.total_episodes, 1),
            'accuracy': self.correct_answer_count / max(self.total_episodes, 1),
        }

    def seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


# 导入re模块用于正则表达式
import re
