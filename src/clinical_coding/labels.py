from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SUBSET_PATH = Path(__file__).resolve().parents[2] / "data" / "icd10cm_subset.csv"


@dataclass(frozen=True)
class Label:
    code: str
    description: str
    domain: str


class UnknownCodeError(ValueError):
    pass


def load_labels(path: Path = DEFAULT_SUBSET_PATH) -> list[Label]:
    with open(path, newline="", encoding="utf-8") as f:
        labels = [Label(row["code"], row["description"], row["domain"]) for row in csv.DictReader(f)]
    codes = [label.code for label in labels]
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate codes in label set")
    return labels


def code_set(labels: list[Label]) -> set[str]:
    return {label.code for label in labels}


def validate_code(code: str, labels: list[Label]) -> None:
    if code not in code_set(labels):
        raise UnknownCodeError(f"{code!r} is not in the closed label set")
