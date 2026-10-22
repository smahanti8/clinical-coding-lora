from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from clinical_coding.labels import load_labels
from clinical_coding.metrics import evaluate
from clinical_coding.parsing import normalize_output
from clinical_coding.prompting import format_prompt
from clinical_coding.split import make_split


@dataclass
class EvalConfig:
    seed: int = 0
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path: str | None = None  # None => zero-shot base model
    n_train: int = 2000
    n_test_id: int = 300
    n_test_ood: int = 300
    max_new_tokens: int = 16
    output_path: str = "results.json"


def data_hash(texts: list[str]) -> str:
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def load_model_for_generation(config: EvalConfig):
    # Heavy imports deferred so this module never requires torch/transformers/
    # peft/bitsandbytes on CPU; only exercised when actually generating.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, quantization_config=bnb_config, device_map="auto"
    )
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path)
    return model, tokenizer


def make_generate_fn(model, tokenizer, config: EvalConfig):
    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,  # greedy decoding, for a reproducible comparison
        )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return generate


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "trl", "bitsandbytes"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = None
        else:
            versions[name] = getattr(module, "__version__", "unknown")
    return versions


def run_eval(config: EvalConfig, generate_fn=None, load_model=load_model_for_generation) -> dict:
    labels = load_labels()
    valid_codes = {label.code for label in labels}
    label_codes = sorted(valid_codes)
    split = make_split(config.seed, config.n_train, config.n_test_id, config.n_test_ood, labels)

    if generate_fn is None:
        model, tokenizer = load_model(config)
        generate_fn = make_generate_fn(model, tokenizer, config)

    results = {}
    for split_name, examples in (
        ("in_distribution", split.test_in_distribution),
        ("heldout_family", split.test_heldout_family),
    ):
        y_true = [e.label for e in examples]
        raw_outputs = [generate_fn(format_prompt(e.text, labels)) for e in examples]
        y_pred = [normalize_output(raw, valid_codes) for raw in raw_outputs]
        results[split_name] = asdict(evaluate(y_true, y_pred, label_codes, seed=config.seed))

    all_texts = [e.text for e in split.train + split.test_in_distribution + split.test_heldout_family]
    return {
        "config": asdict(config),
        "library_versions": _library_versions(),
        "data_hash": data_hash(all_texts),
        "results": results,
    }


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="results.json")
    args = parser.parse_args(argv)
    return EvalConfig(seed=args.seed, adapter_path=args.adapter_path, output_path=args.output_path)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    output = run_eval(config)
    Path(config.output_path).write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
