from clinical_coding.labels import load_labels
from clinical_coding.prompting import format_prompt


def test_prompt_lists_every_valid_code_and_ends_with_the_cue():
    labels = load_labels()
    prompt = format_prompt("Patient presents with cough.", labels)
    assert all(label.code in prompt for label in labels)
    assert prompt.rstrip().endswith("Principal diagnosis code:")


def test_prompt_embeds_the_note_text():
    labels = load_labels()
    prompt = format_prompt("UNIQUE_NOTE_MARKER_12345", labels)
    assert "UNIQUE_NOTE_MARKER_12345" in prompt
