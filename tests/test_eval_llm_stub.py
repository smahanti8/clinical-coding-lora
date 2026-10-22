import sys

HEAVY_MODULES = ("torch", "transformers", "peft", "trl", "bitsandbytes")


def test_importing_module_does_not_pull_in_heavy_libs():
    for mod in HEAVY_MODULES:
        assert mod not in sys.modules
    import scripts.eval_llm  # noqa: F401

    for mod in HEAVY_MODULES:
        assert mod not in sys.modules


def test_run_eval_with_stub_generate_fn_produces_expected_json_schema():
    from clinical_coding.labels import load_labels
    from scripts.eval_llm import EvalConfig, run_eval

    fixed_code = load_labels()[0].code

    def fake_generate(prompt: str) -> str:
        return fixed_code

    config = EvalConfig(seed=1, n_train=20, n_test_id=10, n_test_ood=10)
    output = run_eval(config, generate_fn=fake_generate)

    assert set(output.keys()) == {"config", "library_versions", "data_hash", "results"}
    assert set(output["results"].keys()) == {"in_distribution", "heldout_family"}
    for split_result in output["results"].values():
        for key in ("accuracy", "macro_f1", "invalid_rate", "accuracy_ci95", "confusion_matrix"):
            assert key in split_result
    for mod in HEAVY_MODULES:
        assert mod not in sys.modules


def test_run_eval_reports_full_invalid_rate_for_garbage_generations():
    from scripts.eval_llm import EvalConfig, run_eval

    def fake_generate(prompt: str) -> str:
        return "not a real code"

    config = EvalConfig(seed=2, n_train=20, n_test_id=5, n_test_ood=5)
    output = run_eval(config, generate_fn=fake_generate)

    assert output["results"]["in_distribution"]["invalid_rate"] == 1.0
    assert output["results"]["heldout_family"]["invalid_rate"] == 1.0


def test_data_hash_is_deterministic_for_the_same_seed_and_sizes():
    from scripts.eval_llm import EvalConfig, run_eval

    def fake_generate(prompt: str) -> str:
        return "X"

    config = EvalConfig(seed=3, n_train=10, n_test_id=5, n_test_ood=5)
    a = run_eval(config, generate_fn=fake_generate)
    b = run_eval(config, generate_fn=fake_generate)
    assert a["data_hash"] == b["data_hash"]
