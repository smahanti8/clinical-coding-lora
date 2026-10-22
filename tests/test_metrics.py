import pytest

from clinical_coding.metrics import bootstrap_accuracy_ci, evaluate

LABELS = ["A", "B"]


def test_hand_computed_accuracy_precision_recall_f1():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", None]

    result = evaluate(y_true, y_pred, LABELS, seed=0)

    assert result.accuracy == pytest.approx(0.5)
    assert result.invalid_rate == pytest.approx(0.25)
    assert result.per_code_precision["A"] == pytest.approx(1.0)
    assert result.per_code_precision["B"] == pytest.approx(0.5)
    assert result.per_code_recall["A"] == pytest.approx(0.5)
    assert result.per_code_recall["B"] == pytest.approx(0.5)
    assert result.macro_f1 == pytest.approx((2 / 3 + 0.5) / 2)


def test_confusion_matrix_counts_only_valid_predictions():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", None]

    result = evaluate(y_true, y_pred, LABELS, seed=0)

    assert result.confusion_matrix == {
        "A": {"A": 1, "B": 1},
        "B": {"A": 0, "B": 1},
    }


def test_invalid_rate_with_no_invalid_predictions_is_zero():
    result = evaluate(["A", "B"], ["A", "B"], LABELS, seed=0)
    assert result.invalid_rate == 0.0


def test_bootstrap_ci_contains_point_estimate():
    correct = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    point_estimate = sum(correct) / len(correct)
    lower, upper = bootstrap_accuracy_ci(correct, seed=0, n_bootstrap=500)
    assert lower <= point_estimate <= upper


def test_bootstrap_ci_widens_with_fewer_samples():
    rng_correct_large = [1, 0] * 100
    rng_correct_small = [1, 0]
    lower_large, upper_large = bootstrap_accuracy_ci(rng_correct_large, seed=1, n_bootstrap=1000)
    lower_small, upper_small = bootstrap_accuracy_ci(rng_correct_small, seed=1, n_bootstrap=1000)
    assert (upper_small - lower_small) >= (upper_large - lower_large)


def test_bootstrap_ci_is_deterministic_for_a_given_seed():
    correct = [1, 0, 1, 1, 0]
    a = bootstrap_accuracy_ci(correct, seed=42, n_bootstrap=200)
    b = bootstrap_accuracy_ci(correct, seed=42, n_bootstrap=200)
    assert a == b
