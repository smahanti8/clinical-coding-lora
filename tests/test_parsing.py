from clinical_coding.parsing import normalize_output

VALID = {"E11.9", "I10", "J45.909"}


def test_accepts_code_with_dot():
    assert normalize_output("E11.9", VALID) == "E11.9"


def test_accepts_code_without_dot():
    assert normalize_output("E119", VALID) == "E11.9"


def test_accepts_lowercase():
    assert normalize_output("e11.9", VALID) == "E11.9"


def test_accepts_surrounding_whitespace():
    assert normalize_output("  I10  \n", VALID) == "I10"


def test_rejects_garbage_string():
    assert normalize_output("the code is E11.9", VALID) is None


def test_rejects_unknown_code():
    assert normalize_output("Z99.9", VALID) is None


def test_rejects_empty_string():
    assert normalize_output("", VALID) is None


def test_rejects_none_input():
    assert normalize_output(None, VALID) is None
