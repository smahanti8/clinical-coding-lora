from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

from clinical_coding.labels import load_labels
from clinical_coding.prompting import format_prompt
from clinical_coding.split import make_split

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Verified against microsoft/Phi-3-mini-4k-instruct's modeling_phi3.py:
# Phi3Attention defines qkv_proj/o_proj, Phi3MLP defines gate_up_proj/down_proj.
LORA_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


@dataclass
class TrainConfig:
    seed: int = 0
    model_name: str = MODEL_NAME
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    n_train: int = 2000
    n_test_id: int = 300
    n_test_ood: int = 300
    output_dir: str = "outputs/adapter"
    max_seq_length: int = 512


def build_training_examples(config: TrainConfig) -> list[dict]:
    labels = load_labels()
    split = make_split(config.seed, config.n_train, config.n_test_id, config.n_test_ood, labels)
    return [{"prompt": format_prompt(e.text, labels), "completion": e.label} for e in split.train]


def load_base_model_and_tokenizer(config: TrainConfig):
    # Heavy imports are deferred here so this module (and its logic tests)
    # never require torch/transformers/bitsandbytes on CPU.
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
    return model, tokenizer


def build_lora_config(config: TrainConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )


def _default_train_fn(config: TrainConfig, model, tokenizer, examples: list[dict]) -> None:
    from peft import get_peft_model
    from trl import SFTConfig, SFTTrainer

    model = get_peft_model(model, build_lora_config(config))
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        seed=config.seed,
        max_seq_length=config.max_seq_length,
    )
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=examples, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(config.output_dir)


def run_training(
    config: TrainConfig,
    load_model_and_tokenizer=load_base_model_and_tokenizer,
    train_fn=None,
) -> None:
    random.seed(config.seed)
    examples = build_training_examples(config)
    model, tokenizer = load_model_and_tokenizer(config)
    (train_fn or _default_train_fn)(config, model, tokenizer, examples)


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="outputs/adapter")
    args = parser.parse_args(argv)
    return TrainConfig(seed=args.seed, n_train=args.n_train, output_dir=args.output_dir)


def main(argv: list[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
