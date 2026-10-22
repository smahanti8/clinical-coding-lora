# Running the QLoRA fine-tune on Google Colab (free T4)

This project's GPU scripts (`scripts/train_qlora.py`, `scripts/eval_llm.py`)
are never executed in the CPU development environment - only their logic is
tested there, against a stub model. These are the exact steps to run them
for real on a free Colab T4 GPU. Expected runtime is **unmeasured** - this
repository states no wall-clock estimate because it has not been run; time
it yourself and record it in `results.json` / the README when you do.

## 1. Start a GPU runtime

In Colab: `Runtime -> Change runtime type -> T4 GPU`.

## 2. Clone the repository and install GPU dependencies

```bash
git clone <your-repo-url> clinical-coding-lora
cd clinical-coding-lora
pip install -r requirements.txt
pip install -r requirements-train.txt
```

A T4 has 16GB of VRAM, which is enough for Phi-3-mini-4k-instruct at 4-bit
NF4 quantization provided the sequence length is kept short (this project's
default `max_seq_length=512` is deliberately conservative for this reason).

## 3. Run training

Adapter weights must be saved **outside the repository clone** - the repo's
`.gitignore` already excludes `outputs/` and `adapters/`, but on Colab the
safer pattern is to save straight to Google Drive so the weights survive a
runtime disconnect:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
python scripts/train_qlora.py \
  --seed 0 \
  --n-train 2000 \
  --output-dir /content/drive/MyDrive/clinical-coding-lora-adapter
```

## 4. Run evaluation (zero-shot base model)

```bash
python scripts/eval_llm.py \
  --seed 0 \
  --output-path /content/drive/MyDrive/clinical-coding-lora-results/zero_shot.json
```

## 5. Run evaluation (fine-tuned adapter)

```bash
python scripts/eval_llm.py \
  --seed 0 \
  --adapter-path /content/drive/MyDrive/clinical-coding-lora-adapter \
  --output-path /content/drive/MyDrive/clinical-coding-lora-results/fine_tuned.json
```

## 6. Copy the results JSON back into the repository

Both files under `.../clinical-coding-lora-results/` on Drive are the
artifacts to bring back - copy them to your local clone (e.g. via the Drive
web UI, `rclone`, or `drive.mount` + `cp` in a Colab cell) and paste their
`accuracy`, `macro_f1`, `invalid_rate` and `accuracy_ci95` values into the
results table in `README.md`, replacing the "not run yet" rows. Do not copy
the adapter weights themselves into the repository - only the small JSON
results files.
