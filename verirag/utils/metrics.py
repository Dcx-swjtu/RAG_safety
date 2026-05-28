"""Evaluation metrics used by VeriRAG scripts."""

from dataclasses import dataclass


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def compute_accuracy(correct: int, total: int) -> float:
    """Clean-query answer accuracy."""
    return _safe_divide(correct, total)


def compute_asr(attack_succeeded: int, total_attacks: int) -> float:
    """Attack Success Rate."""
    return _safe_divide(attack_succeeded, total_attacks)


def compute_f1(acc: float, asr: float) -> float:
    """Harmonic mean of clean accuracy and defense success rate."""
    defense_success_rate = 1.0 - asr
    denom = acc + defense_success_rate
    if denom <= 1e-12:
        return 0.0
    return 2.0 * acc * defense_success_rate / denom


def compute_dacc(tp: int, tn: int, total: int) -> float:
    """Detection accuracy."""
    return _safe_divide(tp + tn, total)


def compute_fpr(fp: int, total_clean: int) -> float:
    """False positive rate."""
    return _safe_divide(fp, total_clean)


def compute_fnr(fn: int, total_attacks: int) -> float:
    """False negative rate."""
    return _safe_divide(fn, total_attacks)


def compute_dr(asr_no_defense: float, asr_with_defense: float) -> float:
    """Defense rate relative to a no-defense baseline."""
    if asr_no_defense <= 1e-12:
        return 0.0
    return (asr_no_defense - asr_with_defense) / asr_no_defense


@dataclass
class EvaluationCounts:
    """Counters for computing VeriRAG benchmark metrics."""

    clean_correct: int = 0
    clean_total: int = 0
    attack_succeeded: int = 0
    attack_total: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def to_metrics(self) -> dict:
        acc = compute_accuracy(self.clean_correct, self.clean_total)
        asr = compute_asr(self.attack_succeeded, self.attack_total)
        total = self.clean_total + self.attack_total
        return {
            "acc": acc,
            "asr": asr,
            "f1": compute_f1(acc, asr),
            "dacc": compute_dacc(self.true_positive, self.true_negative, total),
            "fpr": compute_fpr(self.false_positive, self.clean_total),
            "fnr": compute_fnr(self.false_negative, self.attack_total),
        }
