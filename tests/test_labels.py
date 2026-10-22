import pytest

from clinical_coding.labels import UnknownCodeError, load_labels, validate_code


def test_loads_expected_row_count():
    labels = load_labels()
    assert len(labels) == 27


def test_codes_are_unique():
    labels = load_labels()
    codes = [label.code for label in labels]
    assert len(codes) == len(set(codes))


def test_descriptions_are_non_empty():
    labels = load_labels()
    assert all(label.description.strip() for label in labels)


def test_three_domains_present():
    labels = load_labels()
    assert {label.domain for label in labels} == {"endocrine", "respiratory", "circulatory"}


def test_validate_known_code_does_not_raise():
    labels = load_labels()
    validate_code("I10", labels)


def test_validate_unknown_code_raises():
    labels = load_labels()
    with pytest.raises(UnknownCodeError):
        validate_code("Z99.9", labels)
