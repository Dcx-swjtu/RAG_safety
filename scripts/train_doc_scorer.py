#!/usr/bin/env python3
"""Train a supervised adversarial document scorer from aligned attack files."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verirag.learned_doc_scorer import (  # noqa: E402
    AdversarialDocClassifier,
    LearnedAdversarialDocScorer,
    build_doc_classifier_examples,
)
from verirag.text_features import TextFeatureExtractor  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_features(scorer: LearnedAdversarialDocScorer, examples) -> np.ndarray:
    rows: List[np.ndarray] = []
    for ex in examples:
        rows.append(scorer.build_features(ex.query, [ex.doc])[0])
    return np.stack(rows, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned adversarial doc scorer")
    parser.add_argument("--data-dir", default="data_official_nq_split")
    parser.add_argument("--datasets", nargs="+", default=["nq"])
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument(
        "--attack-types",
        nargs="+",
        default=["poisonedrag", "oneshot", "refinerag", "semantic_chameleon", "adaptive"],
    )
    parser.add_argument("--output", default="experiments/doc_scorer/learned_doc_scorer.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-clean-docs-per-sample", type=int, default=10)
    parser.add_argument("--max-attack-docs-per-sample", type=int, default=10)
    parser.add_argument("--embedding-model-path", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = TextFeatureExtractor(
        {
            "embedding_model_path": args.embedding_model_path,
            "max_docs": args.max_clean_docs_per_sample,
        }
    )
    scorer = LearnedAdversarialDocScorer(
        {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "ensemble_weight": 1.0,
            "device": str(device),
        },
        feature_extractor=feature_extractor,
    )

    examples = build_doc_classifier_examples(
        data_dir=args.data_dir,
        datasets=args.datasets,
        attack_types=args.attack_types,
        max_clean_docs_per_sample=args.max_clean_docs_per_sample,
        max_attack_docs_per_sample=args.max_attack_docs_per_sample,
        splits=args.splits,
    )
    if not examples:
        raise RuntimeError(f"No training examples found under {args.data_dir}")

    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    train_examples = examples[:split]
    valid_examples = examples[split:] or examples[: min(len(examples), 128)]

    print(f"[DocScorer] examples={len(examples)} train={len(train_examples)} valid={len(valid_examples)}")
    print(f"[DocScorer] positives={sum(ex.label for ex in examples)} negatives={sum(1 - ex.label for ex in examples)}")
    print(f"[DocScorer] feature_backend={feature_extractor.backend} input_dim={scorer.input_dim}")

    x_train = batch_features(scorer, train_examples)
    y_train = np.asarray([ex.label for ex in train_examples], dtype=np.float32)
    x_valid = batch_features(scorer, valid_examples)
    y_valid = np.asarray([ex.label for ex in valid_examples], dtype=np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = AdversarialDocClassifier(scorer.input_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    best_valid_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)

        model.eval()
        with torch.no_grad():
            valid_logits = model(torch.from_numpy(x_valid).to(device))
            valid_probs = torch.sigmoid(valid_logits).cpu().numpy()
        pred = valid_probs >= 0.5
        labels = y_valid.astype(bool)
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        tn = int((~pred & ~labels).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        acc = (tp + tn) / max(len(labels), 1)
        print(
            f"[DocScorer] epoch={epoch} loss={total_loss / max(len(train_examples), 1):.4f} "
            f"valid_acc={acc:.4f} valid_f1={f1:.4f} precision={precision:.4f} recall={recall:.4f}"
        )
        if f1 > best_valid_f1:
            best_valid_f1 = f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_dim": scorer.input_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "embedding_dim": feature_extractor.embedding_dim,
            },
            "metadata": {
                "data_dir": args.data_dir,
                "datasets": args.datasets,
                "attack_types": args.attack_types,
                "splits": args.splits,
                "num_examples": len(examples),
                "best_valid_f1": best_valid_f1,
                "feature_backend": feature_extractor.backend,
            },
        },
        args.output,
    )
    summary_path = str(Path(args.output).with_suffix(".json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best_valid_f1": best_valid_f1, "num_examples": len(examples)}, f, indent=2)
    print(f"[DocScorer] saved={args.output}")
    print(f"[DocScorer] summary={summary_path}")


if __name__ == "__main__":
    main()
