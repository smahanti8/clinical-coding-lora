from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    per_code_precision: dict[str, float]
    per_code_recall: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    invalid_rate: float
    accuracy_ci95: tuple[float, float]


def bootstrap_accuracy_ci(
    correct: list[int],
    seed: int = 0,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    n = len(correct)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = sorted(
        sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_bootstrap)
    )
    lower_idx = int((alpha / 2) * n_bootstrap)
    upper_idx = min(int((1 - alpha / 2) * n_bootstrap), n_bootstrap - 1)
    return (samples[lower_idx], samples[upper_idx])


def evaluate(
    y_true: list[str],
    y_pred: list[str | None],
    labels: list[str],
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> EvaluationResult:
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    invalid_rate = sum(1 for p in y_pred if p is None) / n if n else 0.0
    correct = [1 if (p is not None and p == t) else 0 for t, p in zip(y_true, y_pred)]
    accuracy = sum(correct) / n if n else 0.0

    confusion = {t: {label: 0 for label in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if p is not None and p in labels and t in labels:
            confusion[t][p] += 1

    support = Counter(y_true)
    per_code_precision: dict[str, float] = {}
    per_code_recall: dict[str, float] = {}
    for code in labels:
        true_positive = confusion[code][code]
        total_true = support[code]
        false_positive = sum(confusion[t][code] for t in labels) - true_positive
        per_code_precision[code] = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )
        per_code_recall[code] = true_positive / total_true if total_true > 0 else 0.0

    f1_scores = []
    for code in labels:
        p, r = per_code_precision[code], per_code_recall[code]
        f1_scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    macro_f1 = sum(f1_scores) / len(labels) if labels else 0.0

    accuracy_ci95 = bootstrap_accuracy_ci(correct, seed=seed, n_bootstrap=n_bootstrap)

    return EvaluationResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_code_precision=per_code_precision,
        per_code_recall=per_code_recall,
        confusion_matrix=confusion,
        invalid_rate=invalid_rate,
        accuracy_ci95=accuracy_ci95,
    )
