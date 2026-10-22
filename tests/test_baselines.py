from collections import Counter

from clinical_coding.baselines import TfidfLogisticRegressionBaseline, majority_class_predict
from clinical_coding.split import make_split


def test_majority_baseline_reproduces_true_majority_class():
    split = make_split(seed=1, n_train=300, n_test_id=100, n_test_ood=100)
    train_labels = [e.label for e in split.train]
    preds = majority_class_predict(train_labels, n=len(split.test_in_distribution))
    expected_majority = Counter(train_labels).most_common(1)[0][0]
    assert all(p == expected_majority for p in preds)


def test_tfidf_logreg_beats_or_matches_majority_on_in_distribution_test():
    split = make_split(seed=2, n_train=1000, n_test_id=300, n_test_ood=300)
    train_texts = [e.text for e in split.train]
    train_labels = [e.label for e in split.train]
    test_texts = [e.text for e in split.test_in_distribution]
    test_labels = [e.label for e in split.test_in_distribution]

    baseline = TfidfLogisticRegressionBaseline.fit(train_texts, train_labels, seed=0)
    preds = baseline.predict(test_texts)
    tfidf_accuracy = sum(p == t for p, t in zip(preds, test_labels)) / len(test_labels)

    majority_label = Counter(train_labels).most_common(1)[0][0]
    majority_accuracy = sum(t == majority_label for t in test_labels) / len(test_labels)

    assert tfidf_accuracy >= majority_accuracy
