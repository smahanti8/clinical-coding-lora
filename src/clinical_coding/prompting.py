from __future__ import annotations

from clinical_coding.labels import Label

# Shared between scripts/train_qlora.py and scripts/eval_llm.py so the
# zero-shot base model and the fine-tuned adapter are ever only compared
# under the exact same prompt.
PROMPT_TEMPLATE = (
    "You are a clinical coding assistant. Read the discharge note and output "
    "ONLY the single ICD-10-CM code for the principal diagnosis, chosen from "
    "the list below. Do not output any other text.\n\n"
    "Valid codes:\n{code_list}\n\n"
    "Note:\n{note}\n\n"
    "Principal diagnosis code:"
)


def format_prompt(note_text: str, labels: list[Label]) -> str:
    code_list = "\n".join(f"{label.code}: {label.description}" for label in labels)
    return PROMPT_TEMPLATE.format(code_list=code_list, note=note_text)
