"""
数据准备脚本

功能:
- 从原始数据集构建训练/验证/测试划分
- 构建声明提取训练数据 (Type A)
- 构建RL轨迹数据 (Type B)
- 构建攻击模拟数据 (Type C)
- 构建评估基准数据 (Type D)
- 数据质量验证

使用方法:
    python scripts/prepare_data.py --config configs/config.yaml --output ./data

参数:
    --config: 配置文件路径
    --output: 输出目录
    --datasets: 源数据集列表
    --phases: 处理阶段
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verirag.claim_extractor import ClaimExtractor, Claim, ClaimType
from verirag.attack_simulator import AttackSimulator


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_dataset(dataset_name: str, data_dir: str, num_mock_samples: int = 1000) -> List[Dict]:
    """
    加载原始数据集

    支持的数据集:
    - NQ (Natural Questions)
    - HotpotQA
    - MS-MARCO
    - 模拟数据
    """
    dataset_path = os.path.join(data_dir, 'raw', f'{dataset_name}.jsonl')

    data = []
    if os.path.exists(dataset_path):
        print(f"[DataPrep] 加载数据集: {dataset_name}")
        with open(dataset_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    else:
        print(f"[DataPrep] 原始数据集未找到: {dataset_path}")
        print(f"[DataPrep] 生成模拟数据: {dataset_name}")
        data = generate_mock_dataset(dataset_name, num_samples=num_mock_samples)

    return data


def generate_mock_dataset(dataset_name: str, num_samples: int = 1000) -> List[Dict]:
    """生成模拟数据集"""
    data = []

    templates = [
        {
            'question': 'What is the capital of France?',
            'answer': 'Paris',
            'target_answer': 'London',
            'documents': [
                'Paris is the capital and most populous city of France.',
                'France is a country in Western Europe with Paris as its capital.',
                'The city of Paris has been the capital of France since 508 AD.',
            ],
        },
        {
            'question': 'Who invented the telephone?',
            'answer': 'Alexander Graham Bell',
            'target_answer': 'Thomas Edison',
            'documents': [
                'Alexander Graham Bell is credited with inventing the first practical telephone.',
                'The telephone was invented by Alexander Graham Bell in 1876.',
                'Bell received the first US patent for the telephone in 1876.',
            ],
        },
        {
            'question': 'What is the largest planet?',
            'answer': 'Jupiter',
            'target_answer': 'Saturn',
            'documents': [
                'Jupiter is the largest planet in the Solar System.',
                'With a diameter of 139,820 km, Jupiter is the largest planet.',
                'Jupiter is more than twice as massive as all other planets combined.',
            ],
        },
    ]

    for i in range(num_samples):
        template = templates[i % len(templates)]
        item = {
            'id': f'{dataset_name}_{i}',
            'question': template['question'],
            'answer': template['answer'],
            'target_answer': template['target_answer'],
            'documents': template['documents'],
            'metadata': {
                'dataset': dataset_name,
                'difficulty': ['easy', 'medium', 'hard'][i % 3],
                'type': ['entity', 'numerical', 'temporal'][i % 3],
            },
        }
        data.append(item)

    return data



def _normalize_runtime_documents(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert string documents to the dict format consumed by evaluation/defense."""
    raw_docs = item.get('documents', [])
    normalized = []
    for idx, doc in enumerate(raw_docs):
        if isinstance(doc, dict):
            text = str(doc.get('text', doc.get('content', doc.get('document', ''))))
            doc_id = str(doc.get('doc_id', doc.get('id', f"{item.get('id', 'sample')}_doc_{idx}")))
            source = doc.get('source', doc.get('metadata', {}).get('source', 'synthetic'))
        else:
            text = str(doc)
            doc_id = f"{item.get('id', 'sample')}_doc_{idx}"
            source = item.get('metadata', {}).get('dataset', 'synthetic')
        normalized.append({'doc_id': doc_id, 'text': text, 'source': source})
    return normalized


def _to_runtime_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """Create one canonical sample for train/evaluate scripts."""
    documents = _normalize_runtime_documents(item)
    question = item.get('question', item.get('query', ''))
    answer = item.get('answer', item.get('ground_truth', ''))
    joined_document = "\n".join(doc['text'] for doc in documents)
    return {
        'id': item.get('id', ''),
        'question': question,
        'query': question,
        'answer': answer,
        'ground_truth': answer,
        'target_answer': item.get('target_answer', ''),
        'documents': documents,
        'document': item.get('document', joined_document),
        'text': item.get('text', joined_document),
        'metadata': item.get('metadata', {}),
    }


def save_standard_splits(dataset_name: str, splits: Dict[str, List[Dict]], output_dir: str) -> Dict[str, int]:
    """Write {dataset}_{train,validation,test}.jsonl files used by train/evaluate."""
    os.makedirs(output_dir, exist_ok=True)
    counts = {}
    for split_name, rows in splits.items():
        output_path = os.path.join(output_dir, f'{dataset_name}_{split_name}.jsonl')
        runtime_rows = [_to_runtime_sample(row) for row in rows]
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in runtime_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        counts[f'{dataset_name}_{split_name}'] = len(runtime_rows)
        print(f"[DataPrep] 标准{split_name} split: {len(runtime_rows)}条 -> {output_path}")
    return counts

def split_dataset(data: List[Dict], ratios: List[float] = [0.8, 0.1, 0.1], seed: int = 42) -> Dict[str, List[Dict]]:
    """
    划分数据集

    Args:
        data: 数据列表
        ratios: [train, val, test] 比例
        seed: 随机种子

    Returns:
        {'train': [...], 'validation': [...], 'test': [...]}
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(data))

    n_train = int(len(data) * ratios[0])
    n_val = int(len(data) * ratios[1])

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        'train': [data[i] for i in train_idx],
        'validation': [data[i] for i in val_idx],
        'test': [data[i] for i in test_idx],
    }

    print(f"[DataPrep] 数据集划分: Train={len(splits['train'])}, "
          f"Val={len(splits['validation'])}, Test={len(splits['test'])}")

    return splits


def build_type_a_claim_extraction_data(data: List[Dict], output_dir: str):
    """
    构建Type A: 声明提取训练数据

    从文档中提取声明作为训练样本
    """
    print("[DataPrep] 构建Type A: Claim Extraction数据")

    claim_extractor = ClaimExtractor()

    claim_samples = []
    for item in data:
        documents = item.get('documents', [])
        for doc_text in documents:
            claims = claim_extractor.extract([doc_text])
            for claim in claims:
                sample = {
                    'doc_id': item.get('id', ''),
                    'doc_text': doc_text,
                    'claim': {
                        'subject': claim.subject,
                        'predicate': claim.predicate,
                        'object': claim.object,
                        'value': claim.value,
                        'claim_type': claim.claim_type.value,
                        'confidence': claim.confidence,
                    },
                }
                claim_samples.append(sample)

    output_path = os.path.join(output_dir, 'type_a_claims.jsonl')
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in claim_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"[DataPrep] Type A数据: {len(claim_samples)}条 -> {output_path}")


def build_type_b_rl_trajectory_data(data: List[Dict], output_dir: str):
    """
    构建Type B: RL轨迹训练数据

    使用warm-start规则生成(s, a, r)序列
    """
    print("[DataPrep] 构建Type B: RL Trajectory数据")

    trajectories = []
    for item in data:
        # 模拟trajectory
        query = item['question']
        answer = item['answer']

        # 简单策略: 根据查询特征决定动作
        state_features = {
            'query_length': len(query.split()),
            'has_numerical': any(c.isdigit() for c in query),
            'has_temporal': any(w in query.lower() for w in ['when', 'year', 'date']),
        }

        # 规则决定动作
        if state_features['has_numerical']:
            action = 2  # DEEP
        elif state_features['has_temporal']:
            action = 1  # LIGHT
        else:
            action = 0  # SKIP

        # 模拟奖励
        reward = 5.0 if action < 3 else -1.0

        trajectory = {
            'query': query,
            'state_features': state_features,
            'action': action,
            'reward': reward,
            'answer': answer,
        }
        trajectories.append(trajectory)

    output_path = os.path.join(output_dir, 'type_b_trajectories.jsonl')

    with open(output_path, 'w') as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')

    print(f"[DataPrep] Type B数据: {len(trajectories)}条 -> {output_path}")


def build_type_c_attack_simulation_data(data: List[Dict], output_dir: str, config: Dict):
    """
    构建Type C: 攻击模拟数据

    使用AttackSimulator生成投毒文档
    """
    print("[DataPrep] 构建Type C: Attack Simulation数据")

    attack_simulator = AttackSimulator(config=config.get('attack_simulator', {}))

    attack_types = ['poisonedrag', 'oneshot', 'refinerag', 'semantic_chameleon']
    attack_samples = []

    for item in data:
        query = item['question']
        target_answer = item.get('target_answer', '')
        original_docs = item.get('documents', [])

        for attack_type in attack_types:
            try:
                poisoned_docs = attack_simulator.generate(
                    query=query,
                    target_answer=target_answer,
                    attack_type=attack_type,
                    original_docs=original_docs,
                )

                sample = {
                    'query': query,
                    'ground_truth': item['answer'],
                    'target_answer': target_answer,
                    'attack_type': attack_type,
                    'original_documents': original_docs,
                    'poisoned_documents': poisoned_docs,
                }
                attack_samples.append(sample)

            except Exception as e:
                print(f"[DataPrep] 攻击生成失败: {e}")

    output_path = os.path.join(output_dir, 'type_c_attacks.jsonl')

    with open(output_path, 'w') as f:
        for sample in attack_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"[DataPrep] Type C数据: {len(attack_samples)}条 -> {output_path}")


def build_type_d_evaluation_benchmark(data: List[Dict], output_dir: str):
    """
    构建Type D: 评估基准数据

    标准化评估集，对齐PoisonedRAG
    """
    print("[DataPrep] 构建Type D: Evaluation Benchmark数据")

    benchmark = []
    for item in data:
        # 标准化格式
        sample = _to_runtime_sample(item)
        benchmark.append(sample)

    # 保存
    output_path = os.path.join(output_dir, 'type_d_benchmark.jsonl')

    with open(output_path, 'w') as f:
        for sample in benchmark:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 生成评估配置
    eval_config = {
        'dataset': 'benchmark',
        'num_questions': len(benchmark),
        'attack_types': ['poisonedrag', 'oneshot', 'refinerag', 'semantic_chameleon', 'adaptive'],
        'n_attacks_per_question': 5,
        'seeds': [42, 43, 44, 45, 46],
    }

    eval_config_path = os.path.join(output_dir, 'eval_config.json')
    with open(eval_config_path, 'w') as f:
        json.dump(eval_config, f, indent=2)

    print(f"[DataPrep] Type D数据: {len(benchmark)}条 -> {output_path}")
    print(f"[DataPrep] 评估配置 -> {eval_config_path}")


def validate_data(data: List[Dict], data_type: str) -> Dict[str, Any]:
    """
    数据质量验证

    检查:
    - 字段完整性
    - 数据格式
    - 重复检测
    """
    print(f"[DataPrep] 验证数据: {data_type}")

    issues = []
    seen_ids = set()
    duplicates = 0

    for item in data:
        # 检查重复
        item_id = item.get('id', '')
        if item_id in seen_ids:
            duplicates += 1
        seen_ids.add(item_id)

        # 检查必要字段
        if 'question' not in item and 'query' not in item:
            issues.append(f"缺少问题字段: {item_id}")

    stats = {
        'total_items': len(data),
        'duplicate_items': duplicates,
        'missing_fields': len(issues),
        'issues': issues[:10],  # 只显示前10个
        'is_valid': len(issues) == 0 and duplicates == 0,
    }

    print(f"[DataPrep] 验证结果: 总计={stats['total_items']}, "
          f"重复={stats['duplicate_items']}, 问题={stats['missing_fields']}")

    return stats


def generate_data_statistics(data_dir: str):
    """生成数据统计报告"""
    stats = {}

    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if filename.endswith('.jsonl') and os.path.isfile(path):
            stats[filename[:-6]] = sum(1 for _ in open(path, encoding='utf-8'))

    stats_path = os.path.join(data_dir, 'data_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n[DataPrep] 数据统计:")
    for data_type, count in stats.items():
        print(f"  {data_type}: {count}条")


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='Prepare Training Data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default='./data',
                        help='输出目录')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['nq', 'hotpotqa'],
                        help='源数据集列表')
    parser.add_argument('--phases', type=str, nargs='+',
                        default=['all'],
                        help='处理阶段 (type_a, type_b, type_c, type_d, all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--mock-samples', type=int, default=1000,
                        help='每个缺失原始数据集生成的模拟样本数')
    args = parser.parse_args()

    print("=" * 60)
    print("数据准备")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(args.seed)

    # 加载配置
    config = load_config(args.config)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 确定处理阶段
    phases = args.phases
    if 'all' in phases:
        phases = ['type_a', 'type_b', 'type_c', 'type_d']

    # 加载数据，并为每个数据集写出 train/validation/test 标准 split
    all_data = []
    split_ratios = config.get('data', {}).get('split_ratios', [0.8, 0.1, 0.1])
    for dataset_name in args.datasets:
        data_dir = config.get('data', {}).get('data_dir', './data')
        data = load_raw_dataset(dataset_name, data_dir, num_mock_samples=args.mock_samples)
        dataset_splits = split_dataset(data, split_ratios, args.seed)
        save_standard_splits(dataset_name, dataset_splits, args.output)
        all_data.extend(data)

    print(f"[DataPrep] 总数据: {len(all_data)}条")

    # 聚合划分用于 Type A-D 辅助文件
    splits = split_dataset(all_data, split_ratios, args.seed)

    # 处理各阶段
    if 'type_a' in phases:
        build_type_a_claim_extraction_data(splits['train'], args.output)

    if 'type_b' in phases:
        build_type_b_rl_trajectory_data(splits['train'], args.output)

    if 'type_c' in phases:
        build_type_c_attack_simulation_data(splits['train'], args.output, config)

    if 'type_d' in phases:
        build_type_d_evaluation_benchmark(splits['test'], args.output)

    # 验证数据
    for data_type in phases:
        path = os.path.join(args.output, f'{data_type}.jsonl')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = [json.loads(line) for line in f]
            validate_data(data, data_type)

    # 生成统计
    generate_data_statistics(args.output)

    print("\n" + "=" * 60)
    print("数据准备完成!")
    print(f"输出目录: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
