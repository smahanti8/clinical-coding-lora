import sys

HEAVY_MODULES = ("torch", "transformers", "peft", "trl", "bitsandbytes")


def test_importing_module_does_not_pull_in_heavy_libs():
    for mod in HEAVY_MODULES:
        assert mod not in sys.modules
    import scripts.train_qlora  # noqa: F401

    for mod in HEAVY_MODULES:
        assert mod not in sys.modules


def test_default_lora_target_modules_match_verified_phi3_architecture():
    from scripts.train_qlora import TrainConfig

    config = TrainConfig()
    assert config.lora_target_modules == ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def test_build_training_examples_uses_shared_prompt_and_closed_labels():
    from clinical_coding.labels import load_labels
    from scripts.train_qlora import TrainConfig, build_training_examples

    config = TrainConfig(seed=1, n_train=20, n_test_id=5, n_test_ood=5)
    examples = build_training_examples(config)
    valid_codes = {label.code for label in load_labels()}

    assert len(examples) == 20
    assert all(ex["completion"] in valid_codes for ex in examples)
    assert all("Principal diagnosis code:" in ex["prompt"] for ex in examples)


def test_run_training_exercises_full_pipeline_with_stub_model_and_train_fn():
    from scripts.train_qlora import TrainConfig, run_training

    calls = {}

    def fake_loader(config):
        return ("stub-model", "stub-tokenizer")

    def fake_train_fn(config, model, tokenizer, examples):
        calls["model"] = model
        calls["tokenizer"] = tokenizer
        calls["n_examples"] = len(examples)

    config = TrainConfig(seed=2, n_train=10, n_test_id=2, n_test_ood=2)
    run_training(config, load_model_and_tokenizer=fake_loader, train_fn=fake_train_fn)

    assert calls["model"] == "stub-model"
    assert calls["tokenizer"] == "stub-tokenizer"
    assert calls["n_examples"] == 10
    for mod in HEAVY_MODULES:
        assert mod not in sys.modules
