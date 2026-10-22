from __future__ import annotations

from dataclasses import dataclass

from clinical_coding.generator import Example, generate_dataset
from clinical_coding.labels import Label, load_labels

# Families seen during training vs. families withheld entirely for testing.
# The split is by phrasing style, not by row, so the held-out set exposes
# memorisation of template artefacts rather than genuine transfer.
TRAIN_FAMILIES = ("direct", "abbrev", "narrative", "history_of")
HELDOUT_FAMILIES = ("plan", "chief_complaint")


@dataclass(frozen=True)
class DatasetSplit:
    train: list[Example]
    test_in_distribution: list[Example]
    test_heldout_family: list[Example]


def make_split(
    seed: int,
    n_train: int,
    n_test_id: int,
    n_test_ood: int,
    labels: list[Label] | None = None,
) -> DatasetSplit:
    labels = labels if labels is not None else load_labels()
    train = generate_dataset(seed, n_train, labels, list(TRAIN_FAMILIES))
    test_in_distribution = generate_dataset(seed + 1, n_test_id, labels, list(TRAIN_FAMILIES))
    test_heldout_family = generate_dataset(seed + 2, n_test_ood, labels, list(HELDOUT_FAMILIES))
    return DatasetSplit(
        train=train,
        test_in_distribution=test_in_distribution,
        test_heldout_family=test_heldout_family,
    )
