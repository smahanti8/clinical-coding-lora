from __future__ import annotations


def _lookup_table(valid_codes: set[str]) -> dict[str, str]:
    table: dict[str, str] = {}
    for code in valid_codes:
        upper = code.upper()
        table[upper] = code
        table[upper.replace(".", "")] = code
    return table


def normalize_output(raw: str | None, valid_codes: set[str]) -> str | None:
    """Map a raw model output string to a canonical code, or None if invalid.

    Accepts a code with or without the decimal point and surrounding
    whitespace, case-insensitively. Anything else - including strings that
    merely contain a valid code as a substring - is rejected, since a
    hallucinated code embedded in extra text is still a hallucination.
    """
    if not raw:
        return None
    candidate = raw.strip().upper()
    if not candidate:
        return None
    return _lookup_table(valid_codes).get(candidate)
