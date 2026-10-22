# Key decisions

## ICD-10-CM, not CPT

The label set is drawn entirely from ICD-10-CM, a U.S. government work in
the public domain. CPT is proprietary to the American Medical Association
and cannot be redistributed, so no CPT code or description appears anywhere
in this repository. See `data/PROVENANCE.md` for the exact source file,
fiscal year and SHA-256.

## Label set: 27 codes across 3 domains

Endocrine (Type 2 diabetes mellitus variants), respiratory (pneumonia,
COPD, asthma, bronchitis, respiratory failure) and circulatory
(hypertension, MI, ischemic heart disease, heart failure, atrial
fibrillation/flutter) were chosen because each domain contains several
codes that are genuinely easy to confuse with each other (e.g. the nine
diabetes-with-complication codes), which makes per-code precision/recall
and the confusion matrix meaningful rather than trivially perfect. All 27
codes and descriptions were looked up directly in the downloaded CMS file
(`data/PROVENANCE.md`), not invented or recalled from memory.

## Split by template family, not by row

`src/clinical_coding/split.py` withholds two of six phrasing-style
"families" entirely from training (`plan`, `chief_complaint`) and tests on
them separately from an in-distribution test set drawn from the four
training families. A random row-level split would only measure whether the
model memorised surface strings it already saw; the family split instead
asks whether it generalises to phrasing it has never encountered, which is
closer to (though still far short of) generalising to real clinical
writing style.

## Removing the code string from the generator (a caught leakage bug)

An earlier version of the generator embedded the ICD-10-CM code itself
inside the "abbrev" and "plan" template text (e.g. "Dx: E11.9 - T2DM").
Measuring the TF-IDF baseline against it showed suspiciously high accuracy
on the held-out-family test set - the vectorizer was picking up the code
string as a literal token, which is not a coding task at all. Both
families were rewritten to describe the diagnosis in words only. This is
exactly the kind of template artefact the held-out-family split exists to
catch, and it is recorded here because it shaped the final generator
design, not because it is still present.

## TF-IDF + logistic regression as the CPU baseline

A linear bag-of-words baseline is cheap to run here and, per the README,
must be shown next to any fine-tuned result. If a 4-bit-quantised LoRA
adapter cannot clearly beat a TF-IDF classifier on this synthetic,
template-generated data, that is a meaningful (negative) finding about the
task's difficulty, not a reason to hide the comparison.

## No accuracy or clinical-validity claims

This project measures reproducibility and the mechanics of a QLoRA
fine-tune, not diagnostic accuracy. Every number in the README is either
measured on CPU with a stated seed and sample size, or marked "not run
yet" until a real GPU run produces it.
