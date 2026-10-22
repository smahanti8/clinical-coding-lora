from __future__ import annotations

import random
from dataclasses import dataclass

from clinical_coding.labels import Label, load_labels

FAMILIES = ("direct", "abbrev", "narrative", "history_of", "plan", "chief_complaint")

_OPENERS = (
    "This is a {age}-year-old {sex} admitted to the medical floor.",
    "{sex_title} presents to the emergency department.",
    "Patient is a {age}-year-old {sex} with the following history.",
)

_NEGATIONS = (
    "Denies chest pain.",
    "No shortness of breath at rest.",
    "Denies fever or chills.",
    "No nausea or vomiting reported.",
)

_ABBREVIATIONS = {
    "type 2 diabetes mellitus": "T2DM",
    "chronic obstructive pulmonary disease": "COPD",
    "essential (primary) hypertension": "HTN",
    "acute myocardial infarction": "MI",
    "heart failure": "CHF",
    "atrial fibrillation": "AFib",
}


@dataclass(frozen=True)
class Example:
    text: str
    label: str
    family: str


def _lowered(description: str) -> str:
    return description[0].lower() + description[1:]


def _apply_family(family: str, description: str) -> str:
    desc_lower = _lowered(description)
    if family == "direct":
        return f"Assessment: {description}."
    if family == "abbrev":
        short = description
        for full, abbr in _ABBREVIATIONS.items():
            if full in description.lower():
                short = abbr
                break
        return f"Dx: {short}."
    if family == "narrative":
        return f"The patient is admitted with {desc_lower}."
    if family == "history_of":
        return f"History of present illness notable for {desc_lower}."
    if family == "plan":
        return f"Plan: continue management of {desc_lower}."
    if family == "chief_complaint":
        return f"Chief complaint: {desc_lower}."
    raise ValueError(f"unknown template family: {family!r}")


def generate_example(rng: random.Random, labels: list[Label], families: list[str]) -> Example:
    label = rng.choice(labels)
    family = rng.choice(families)
    opener = rng.choice(_OPENERS).format(
        age=rng.randint(24, 88),
        sex=rng.choice(["male", "female"]),
        sex_title=rng.choice(["He", "She"]),
    )
    principal = _apply_family(family, label.description)
    comorbid_label = rng.choice([other for other in labels if other.code != label.code])
    comorbid_sentence = f"Past medical history significant for {_lowered(comorbid_label.description)}."
    negation_sentence = rng.choice(_NEGATIONS)
    text = " ".join([opener, principal, comorbid_sentence, negation_sentence])
    return Example(text=text, label=label.code, family=family)


def generate_dataset(
    seed: int,
    n: int,
    labels: list[Label] | None = None,
    families: list[str] | None = None,
) -> list[Example]:
    labels = labels if labels is not None else load_labels()
    families = families if families is not None else list(FAMILIES)
    rng = random.Random(seed)
    return [generate_example(rng, labels, families) for _ in range(n)]
