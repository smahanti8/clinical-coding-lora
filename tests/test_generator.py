from clinical_coding.generator import FAMILIES, generate_dataset
from clinical_coding.labels import load_labels


def test_same_seed_is_deterministic():
    a = generate_dataset(seed=7, n=50)
    b = generate_dataset(seed=7, n=50)
    assert [e.text for e in a] == [e.text for e in b]
    assert [e.label for e in a] == [e.label for e in b]


def test_different_seeds_differ():
    a = generate_dataset(seed=1, n=50)
    b = generate_dataset(seed=2, n=50)
    assert [e.text for e in a] != [e.text for e in b]


def test_families_produce_distinct_phrasing_for_same_code():
    labels = load_labels()
    code = labels[0].code
    examples = generate_dataset(seed=3, n=400, labels=labels, families=list(FAMILIES))
    texts_for_code = {e.text for e in examples if e.label == code}
    assert len(texts_for_code) > 1


def test_restricting_families_is_honored():
    subset = ["direct", "abbrev"]
    examples = generate_dataset(seed=4, n=100, families=subset)
    assert {e.family for e in examples} <= set(subset)


def test_all_labels_are_valid_codes():
    labels = load_labels()
    valid_codes = {label.code for label in labels}
    examples = generate_dataset(seed=5, n=200, labels=labels)
    assert all(e.label in valid_codes for e in examples)


def test_negation_sentence_present_and_unrelated_to_any_description():
    labels = load_labels()
    examples = generate_dataset(seed=6, n=50, labels=labels)
    descriptions_lower = [label.description.lower() for label in labels]
    for example in examples:
        assert "denies" in example.text.lower() or "no " in example.text.lower()
        # the negation clause itself must not restate a code description
        negation_clause = example.text.split(".")[-2].strip().lower()
        assert not any(desc in negation_clause for desc in descriptions_lower)
