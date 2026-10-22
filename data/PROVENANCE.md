# Data provenance

## Source file

- Publisher: Centers for Medicare & Medicaid Services (CMS), public domain
  (ICD-10-CM is a U.S. government work; unlike CPT, it may be freely
  redistributed).
- Fiscal year: FY 2026 (effective October 1, 2025 - March 31, 2026).
- Download URL: `https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip`
- Downloaded file size: 2,204,985 bytes
- Downloaded file SHA-256: `55a9124a27ca78a4a5c41e1101eda064ec2e08d105ff9e78d1fbd1a83298ef11`
- Download date: 2026-09-27
- File used from the archive: `icd10cm_codes_2026.txt` ("Code Descriptions in
  Tabular Order" - one code and one short description per line, raw code
  format with no decimal point, e.g. `E119`).

## Subset extraction

`data/icd10cm_subset.csv` contains 27 rows hand-selected from the file above,
spanning three domains (endocrine, respiratory, circulatory) chosen for
template variety and clinically meaningful confusions. Every `code` and
`description` value is copied verbatim from `icd10cm_codes_2026.txt`; the
only transformation applied is inserting the conventional decimal point
after the third character for codes longer than three characters (CMS's raw
order file omits it, e.g. `E119` -> `E11.9`), which does not alter the text
of any description. No CPT codes or descriptions are used anywhere in this
repository.

The raw downloaded zip and extracted `.txt`/`.pdf` files are not committed
(see `.gitignore`); only the 27-row subset actually used by this project is
checked in, per this provenance note.
