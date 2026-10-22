from clinical_coding.labels import load_labels
from clinical_coding.split import HELDOUT_FAMILIES, TRAIN_FAMILIES, make_split


def test_train_and_heldout_families_are_disjoint():
    assert set(TRAIN_FAMILIES).isdisjoint(HELDOUT_FAMILIES)


def test_heldout_family_test_set_never_uses_a_train_family():
    split = make_split(seed=1, n_train=500, n_test_id=100, n_test_ood=100)
    train_families = {e.family for e in split.train}
    ood_families = {e.family for e in split.test_heldout_family}
    assert train_families.isdisjoint(ood_families)


def test_in_distribution_test_only_uses_train_families():
    split = make_split(seed=1, n_train=500, n_test_id=100, n_test_ood=100)
    id_families = {e.family for e in split.test_in_distribution}
    assert id_families <= set(TRAIN_FAMILIES)


def test_every_code_appears_in_train_with_enough_examples():
    labels = load_labels()
    split = make_split(seed=2, n_train=800, n_test_id=100, n_test_ood=100)
    train_codes = {e.label for e in split.train}
    assert train_codes == {label.code for label in labels}


def test_split_is_deterministic_for_a_given_seed():
    a = make_split(seed=9, n_train=200, n_test_id=50, n_test_ood=50)
    b = make_split(seed=9, n_train=200, n_test_id=50, n_test_ood=50)
    assert [e.text for e in a.train] == [e.text for e in b.train]
    assert [e.text for e in a.test_heldout_family] == [e.text for e in b.test_heldout_family]
