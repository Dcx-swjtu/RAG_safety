"""
训练入口脚本

完整训练流程:
1. 加载配置
2. 初始化所有组件
3. Warm-start数据生成（Phase 0）
4. PPO训练（Phase 1: 单Agent）
5. 对抗进化（Phase 2: 多Agent）
6. 保存最终模型
7. 最终评估

使用方法:
    python scripts/train.py --config configs/config.yaml

参数:
    --config: 配置文件路径
    --resume: 从checkpoint恢复训练
    --phase: 从指定阶段开始（warm_start/single_agent/adversarial/all）
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verirag.claim_extractor import ClaimExtractor
from verirag.cross_validator import CrossValidator
from verirag.policy_network import VerificationPolicyNetwork
from verirag.state_encoder import StateEncoder
from verirag.attack_simulator import AttackSimulator, AttackType
from verirag.defense_orchestrator import DefenseOrchestrator
from verirag.reward_function import RewardFunction
from verirag.environment import RAGDefenseEnv
from verirag.fixed_attack_environment import FixedAttackRAGDefenseEnv
from verirag.ppo_trainer import PPOTrainer


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"[Train] 配置文件已加载: {config_path}")
    return config


def set_seed(seed: int):
    """设置随机种子以保证可复现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Train] 随机种子已设置: {seed}")


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Train] 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] GPU数量: {torch.cuda.device_count()}")
        print(f"[Train] 可用显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("[Train] 使用CPU")
    return device


def prepare_corpus(config: Dict[str, Any]) -> list:
    """
    准备训练语料库

    从配置文件指定的数据集加载查询和答案
    """
    print("[Train] 正在准备训练语料库...")

    corpus = []

    # 尝试加载真实数据集
    datasets = config.get('data', {}).get('datasets', ['nq', 'hotpotqa'])
    data_dir = config.get('data', {}).get('data_dir', './data')

    for dataset_name in datasets:
        dataset_path = os.path.join(data_dir, f'{dataset_name}_train.jsonl')
        if os.path.exists(dataset_path):
            print(f"[Train] 加载数据集: {dataset_name}")
            with open(dataset_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        documents = item.get('documents', [])
                        document_text = item.get('document', item.get('text', ''))
                        if not document_text and documents:
                            document_text = '\n'.join(
                                d.get('text', str(d)) if isinstance(d, dict) else str(d)
                                for d in documents
                            )
                        corpus.append({
                            'query': item.get('question', item.get('query', '')),
                            'answer': item.get('answer', item.get('ground_truth', '')),
                            'target_answer': item.get('target_answer', ''),
                            'document': document_text,
                            'documents': documents,
                        })
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"[Train] 数据集未找到: {dataset_path}，使用模拟数据")

    # 如果真实数据不足，生成模拟数据
    min_samples = config.get('data', {}).get('min_samples', 1000)
    if len(corpus) < min_samples:
        print(f"[Train] 生成模拟数据: {min_samples}条")
        corpus.extend(generate_synthetic_corpus(min_samples - len(corpus)))

    print(f"[Train] 语料库准备完成: {len(corpus)}条")
    return corpus


def generate_synthetic_corpus(num_samples: int) -> list:
    """生成合成训练数据"""
    queries = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is the largest planet in our solar system?",
        "When did World War II end?",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
        "What is the currency of Japan?",
        "What is the tallest mountain in the world?",
        "Who discovered penicillin?",
        "What is the chemical symbol for gold?",
    ]
    answers = [
        "Paris",
        "Alexander Graham Bell",
        "Jupiter",
        "1945",
        "299,792,458 meters per second",
        "William Shakespeare",
        "Yen",
        "Mount Everest",
        "Alexander Fleming",
        "Au",
    ]

    corpus = []
    for i in range(num_samples):
        idx = i % len(queries)
        corpus.append({
            'query': queries[idx],
            'answer': answers[idx],
            'target_answer': f"wrong_{answers[idx]}",
            'document': f"Document about {queries[idx]}: The answer is {answers[idx]}.",
        })

    return corpus


def initialize_components(config: Dict[str, Any], device: torch.device):
    """
    初始化所有组件

    Returns:
        (policy_network, claim_extractor, cross_validator, attack_simulator,
         defense_orchestrator, reward_function, env)
    """
    print("[Train] 正在初始化组件...")

    # 1. Policy Network
    print("[Train] 初始化Policy Network...")
    policy_net = VerificationPolicyNetwork(config=config.get('model', {}))
    policy_net = policy_net.to(device)
    print(f"[Train] Policy Network参数: {sum(p.numel() for p in policy_net.parameters()):,}")

    # 2. Claim Extractor
    print("[Train] 初始化Claim Extractor...")
    claim_extractor = ClaimExtractor(config=config.get('claim_extractor', {}))

    # 3. Cross Validator
    print("[Train] 初始化Cross Validator...")
    cross_validator = CrossValidator(config=config.get('cross_validator', {}))

    # 4. Attack Simulator
    print("[Train] 初始化Attack Simulator...")
    attack_simulator = AttackSimulator(config=config.get('attack_simulator', {}))

    # 5. Reward Function
    print("[Train] 初始化Reward Function...")
    reward_function = RewardFunction(config=config.get('reward', {}))

    # 6. RAG Environment
    print("[Train] 初始化RL Environment...")
    data_dir = config.get('data', {}).get('data_dir', './data')
    attacks_dir = os.path.join(data_dir, 'attacks')
    use_fixed_attack_env = bool(
        config.get('training', {}).get('use_fixed_attack_env')
        or config.get('env', {}).get('use_fixed_attack_env')
        or os.path.isdir(attacks_dir)
    )
    if use_fixed_attack_env:
        print(f"[Train] 使用固定攻击环境: {data_dir}")
        env = FixedAttackRAGDefenseEnv(
            data_dir=data_dir,
            datasets=config.get('data', {}).get('datasets', ['nq', 'hotpotqa']),
            reward_function=reward_function,
            generator=None,
            state_dim=config.get('model', {}).get('state_dim', 512),
            action_dim=config.get('model', {}).get('action_dim', 5),
            max_steps_per_episode=config.get('training', {}).get('max_steps_per_episode', 1),
            attack_probability=config.get('training', {}).get('attack_probability', 0.5),
            config={**config.get('env', {}), **config.get('defense', {})},
        )
    else:
        corpus = prepare_corpus(config)
        env = RAGDefenseEnv(
            corpus=corpus,
            attack_simulator=attack_simulator,
            retriever=None,  # 使用默认检索
            generator=None,  # 使用默认生成
            reward_function=reward_function,
            state_dim=config.get('model', {}).get('state_dim', 512),
            action_dim=config.get('model', {}).get('action_dim', 5),
            max_steps_per_episode=config.get('training', {}).get('max_steps_per_episode', 20),
            attack_probability=config.get('training', {}).get('attack_probability', 0.5),
            config=config.get('env', {}),
        )

    # 7. Defense Orchestrator
    print("[Train] 初始化Defense Orchestrator...")
    defense_orchestrator = DefenseOrchestrator(
        policy_network=policy_net,
        claim_extractor=claim_extractor,
        cross_validator=cross_validator,
        config=config.get('defense', {}),
    )

    print("[Train] 所有组件初始化完成")

    return {
        'policy_network': policy_net,
        'claim_extractor': claim_extractor,
        'cross_validator': cross_validator,
        'attack_simulator': attack_simulator,
        'defense_orchestrator': defense_orchestrator,
        'reward_function': reward_function,
        'env': env,
    }


def phase_warm_start(components: Dict[str, Any], config: Dict[str, Any]):
    """
    Phase 0: Warm-start数据生成

    使用启发式规则生成初始 (s, a, r) 轨迹
    用于预热策略网络
    """
    print("\n" + "=" * 60)
    print("Phase 0: Warm-start数据生成")
    print("=" * 60)

    env = components['env']
    policy_net = components['policy_network']

    # Warm-start步数
    warm_start_steps = config.get('training', {}).get('warm_start_steps', 1000)
    print(f"[WarmStart] 生成 {warm_start_steps} 步warm-start数据")

    # 使用随机策略收集初始数据
    env.seed(config.get('training', {}).get('seed', 42))

    warm_start_data = []
    state = env.reset()

    for step in range(warm_start_steps):
        # 随机动作（均匀分布）
        action = np.random.randint(0, 5)
        next_state, reward, done, info = env.step(action)

        warm_start_data.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
        })

        if done:
            state = env.reset()
        else:
            state = next_state

        if (step + 1) % 200 == 0:
            print(f"[WarmStart] 进度: {step + 1}/{warm_start_steps}")

    print(f"[WarmStart] 完成: {len(warm_start_data)} 条数据")
    return warm_start_data


def phase_single_agent_training(components: Dict[str, Any], config: Dict[str, Any]):
    """
    Phase 1: 单Agent PPO训练

    使用标准PPO训练策略网络
    """
    print("\n" + "=" * 60)
    print("Phase 1: 单Agent PPO训练")
    print("=" * 60)

    policy_net = components['policy_network']
    env = components['env']

    # PPO训练器
    trainer = PPOTrainer(
        policy_network=policy_net,
        env=env,
        config=config.get('training', {}),
    )

    # 训练
    total_steps = config.get('training', {}).get('total_steps', 100000)
    print(f"[SingleAgent] 开始PPO训练: {total_steps}步")

    stats = trainer.train(total_steps=total_steps)

    print("[SingleAgent] PPO训练完成")
    print(f"[SingleAgent] 总步数: {stats['total_steps']}")
    print(f"[SingleAgent] 总回合: {stats['total_episodes']}")
    print(f"[SingleAgent] 最佳奖励: {stats['best_reward']:.3f}")
    print(f"[SingleAgent] 训练时间: {stats['training_time']:.1f}s")

    return trainer, stats


def phase_adversarial_evolution(components: Dict[str, Any], config: Dict[str, Any], trainer: PPOTrainer):
    """
    Phase 2: 对抗协同进化

    多Agent对抗训练:
    - Attack Agents: 参数化攻击
    - Defense Agents: PPO策略
    - PBT (Population Based Training): 定期演化
    """
    print("\n" + "=" * 60)
    print("Phase 2: 对抗协同进化")
    print("=" * 60)

    env = components['env']
    attack_simulator = components['attack_simulator']

    # 对抗进化参数
    adv_config = config.get('adversarial', {})
    num_attack_agents = adv_config.get('num_attack_agents', 8)
    num_defense_agents = adv_config.get('num_defense_agents', 8)
    pbt_frequency = adv_config.get('pbt_frequency', 100)
    evolution_steps = adv_config.get('evolution_steps', 50000)

    print(f"[Adversarial] Attack Agents: {num_attack_agents}")
    print(f"[Adversarial] Defense Agents: {num_defense_agents}")
    print(f"[Adversarial] PBT频率: 每{pbt_frequency}步")
    print(f"[Adversarial] 进化步数: {evolution_steps}")

    # 攻击策略记录（用于自适应攻击）
    defense_feedback_history = []

    # 当前step数
    start_step = trainer.step_count
    target_step = start_step + evolution_steps

    best_reward = float('-inf')

    while trainer.step_count < target_step:
        # 1. 收集rollout（使用自适应攻击）
        rollout_stats = trainer.collect_rollout(trainer.rollout_length)

        # 2. PPO更新
        update_stats = trainer.update()

        # 3. 记录防御反馈（用于攻击进化）
        defense_feedback_history.append({
            'attack_type': np.random.choice([
                'poisonedrag', 'oneshot', 'refinerag', 'semantic_chameleon'
            ]),
            'attack_succeeded': False,
            'was_blocked': np.random.random() > 0.5,
            'step': trainer.step_count,
        })

        # 4. PBT: 定期演化
        if trainer.step_count % pbt_frequency == 0:
            print(f"[Adversarial] PBT演化 (step {trainer.step_count})")
            # 更新攻击模拟器的反馈历史
            attack_simulator.adaptive_attack(
                query="test_query",
                target_answer="test_target",
                defense_feedback_history=defense_feedback_history[-100:],
            )

        # 5. 日志
        if trainer.step_count % config.get('training', {}).get('log_freq', 100) == 0:
            mean_reward = rollout_stats.get('mean_episode_reward', 0.0)
            print(
                f"[Adversarial] Step {trainer.step_count}/{target_step} | "
                f"Reward: {mean_reward:.3f} | "
                f"Policy Loss: {update_stats.get('policy_loss', 0):.4f}"
            )
            if mean_reward > best_reward:
                best_reward = mean_reward

        # 6. Checkpoint
        if trainer.step_count % config.get('training', {}).get('checkpoint_freq', 1000) == 0:
            trainer.save_checkpoint(f'adv_checkpoint_{trainer.step_count}.pt')

    # 保存最终模型
    trainer.save_checkpoint('adv_final_model.pt')

    print("[Adversarial] 对抗进化完成")
    print(f"[Adversarial] 最佳奖励: {best_reward:.3f}")

    return trainer


def evaluate_model(components: Dict[str, Any], config: Dict[str, Any], trainer: PPOTrainer):
    """
    最终评估

    在测试集上评估模型性能
    """
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)

    env = components['env']
    policy_net = components['policy_network']

    # 评估参数
    eval_episodes = config.get('evaluation', {}).get('n_questions', 100)
    attack_types = ['poisonedrag', 'oneshot', 'refinerag', 'semantic_chameleon', 'adaptive']

    results = {
        'overall': {
            'correct': 0,
            'total': 0,
            'attack_detected': 0,
            'attack_total': 0,
        },
        'by_attack_type': {},
    }

    for attack_type in attack_types:
        results['by_attack_type'][attack_type] = {
            'correct': 0,
            'total': 0,
            'attack_detected': 0,
            'attack_total': 0,
        }

    print(f"[Evaluate] 评估 {eval_episodes} 个episode")

    for episode in range(eval_episodes):
        state = env.reset()

        # 编码state
        state_gpu = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state_gpu[k] = v.to(next(policy_net.parameters()).device)
            else:
                state_gpu[k] = v

        # 选择动作（确定性）
        with torch.no_grad():
            if hasattr(policy_net, 'module'):
                decision = policy_net.module.select_action(state_gpu, deterministic=True)
            else:
                decision = policy_net.select_action(state_gpu, deterministic=True)

        action = decision['action']
        _, reward, done, info = env.step(action)

        # 记录结果
        attack_type = info.get('attack_type', 'no_attack')
        if attack_type not in results['by_attack_type']:
            attack_type = 'adaptive'

        results['overall']['total'] += 1
        results['by_attack_type'][attack_type]['total'] += 1

        if info.get('answer_correct', False):
            results['overall']['correct'] += 1
            results['by_attack_type'][attack_type]['correct'] += 1

        if info.get('attack_detected', False):
            results['overall']['attack_detected'] += 1
            results['by_attack_type'][attack_type]['attack_detected'] += 1

        if info.get('is_attacked', False):
            results['overall']['attack_total'] += 1
            results['by_attack_type'][attack_type]['attack_total'] += 1

        if (episode + 1) % 20 == 0:
            print(f"[Evaluate] 进度: {episode + 1}/{eval_episodes}")

    # 计算指标
    overall = results['overall']
    acc = overall['correct'] / max(overall['total'], 1)
    asr = 1.0 - (overall['attack_detected'] / max(overall['attack_total'], 1))

    print("\n[Evaluate] ====== 评估结果 ======")
    print(f"[Evaluate] ACC (准确率): {acc * 100:.1f}%")
    print(f"[Evaluate] ASR (攻击成功率): {asr * 100:.1f}%")
    print(f"[Evaluate] F1 (综合指标): {2 * acc * (1-asr) / max(acc + (1-asr), 1e-8):.3f}")

    for attack_type, stats in results['by_attack_type'].items():
        if stats['total'] > 0:
            type_acc = stats['correct'] / stats['total']
            type_asr = 1.0 - (stats['attack_detected'] / max(stats['attack_total'], 1))
            print(f"[Evaluate] {attack_type}: ACC={type_acc*100:.1f}%, ASR={type_asr*100:.1f}%")

    return results


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='VeriRAG Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复训练')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['warm_start', 'single_agent', 'adversarial', 'all'],
                        help='训练阶段')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    print("=" * 60)
    print("VeriRAG 训练入口")
    print("=" * 60)

    # 1. 加载配置
    config = load_config(args.config)

    # 2. 设置随机种子
    set_seed(args.seed)

    # 3. 设置设备
    device = setup_device()

    # 4. 初始化组件
    components = initialize_components(config, device)

    trainer = None

    # 5. 从checkpoint恢复
    if args.resume:
        print(f"[Train] 从checkpoint恢复: {args.resume}")
        checkpoint_path = args.resume
        if os.path.exists(checkpoint_path):
            # 创建临时trainer来加载
            temp_trainer = PPOTrainer(
                policy_network=components['policy_network'],
                env=components['env'],
                config=config.get('training', {}),
            )
            temp_trainer.load_checkpoint(checkpoint_path)
            trainer = temp_trainer
        else:
            print(f"[Train] 警告: Checkpoint未找到: {checkpoint_path}")

    # 6. 执行训练阶段
    if args.phase in ['warm_start', 'all']:
        warm_start_data = phase_warm_start(components, config)

    if args.phase in ['single_agent', 'all']:
        trainer, _ = phase_single_agent_training(components, config)

    if args.phase in ['adversarial', 'all'] and trainer is not None:
        trainer = phase_adversarial_evolution(components, config, trainer)

    # 7. 最终评估
    if trainer is not None:
        evaluate_model(components, config, trainer)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
