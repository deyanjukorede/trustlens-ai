# TrustLens AI Quickstart Tutorial

A beginner-friendly, step-by-step guide to your first TrustLens AI assessment.
No prior knowledge of the project's internals is needed — every command below
was executed and tested against TrustLens AI v1.0.0.

> **What you will do:** install the package, assess a small synthetic CSV
> dataset with the command-line interface, and read the generated JSON report.

## 1. Prerequisites

- Python 3.10, 3.11, or 3.12 (the versions tested in GitHub Actions).
- `pip` available in your terminal.

## 2. Installation

Install the released version from PyPI:

```bash
pip install trustlens-ai==1.0.0
```

Verify the installation:

```bash
trustlens --help
```

To work from a source checkout instead (for example to follow along with the
repository examples), clone and install in editable mode:

```bash
git clone https://github.com/deyanjukorede/trustlens-ai.git
cd trustlens-ai
pip install -e .
```

## 3. The example dataset

The tutorial uses `examples/sample_dataset.csv`, a small **synthetic** dataset
(20 rows, no personal information) with these columns:

| Column | Meaning |
|---|---|
| `record_id` | Unique row identifier |
| `age`, `income` | Example numeric attributes |
| `employment_status`, `region` | Example categorical attributes |
| `group` | Sensitive attribute used for fairness checks |
| `target` | Ground-truth label (`0`/`1`) |
| `prediction` | Example model prediction (`0`/`1`) |

## 4. Run your first assessment

From the repository root (or any directory if you installed from PyPI and have
a copy of the sample CSV), run:

```bash
trustlens assess examples/sample_dataset.csv --output report.json --target target --prediction prediction --sensitive group --identifier record_id
```

What each argument means:

| Argument | Meaning |
|---|---|
| `examples/sample_dataset.csv` | Input CSV to assess (positional) |
| `--output report.json` | Where the JSON report is saved (required) |
| `--target target` | Ground-truth label column |
| `--prediction prediction` | Model prediction column |
| `--sensitive group` | Sensitive-attribute column(s) for fairness checks |
| `--identifier record_id` | Unique row identifier column |

The column names must match the columns in your dataset. On success you see:

```text
TrustLens assessment completed. Report saved to: report.json
```

## 5. Read the JSON report

Open `report.json`. Its top-level sections are:

| Section | Contents |
|---|---|
| `dimensions` | The five assessment dimensions (below), each with findings and evidence |
| `analysis_availability` | Which analyses ran and which were unavailable (with reasons) |
| `limitations` | What the results do and do not mean |
| `cross_dimension_summary` | Findings spanning more than one dimension |
| `assessment_context` | Dataset shape, columns used, and which optional inputs were supplied |

### The five assessment dimensions, in plain language

1. **Data Quality** — is the data complete and well-formed? Missing values,
   duplicates, wrong types, outliers.
2. **Data Governance** — privacy and control signals: sensitive data,
   metadata, ownership, governance controls.
3. **AI Readiness** — can this data train a model? Class imbalance, feature
   suitability, leakage risks.
4. **Responsible AI** — fairness across groups (`group`), bias indicators,
   prediction fairness, explainability readiness.
5. **Operational Trust** — will it hold up over time? Data drift, stability,
   reproducibility and monitoring readiness.

### Review indicators

Look for keys ending in `_requiring_review` (for example
`bias_attributes_requiring_review`, `dimensions_requiring_review`). Each one
names a condition a human should look at — the report supports decisions, it
never replaces human review.

### Unavailable analyses

`analysis_availability` tells you what did **not** run and why. Typical
reasons: no reference CSV was supplied (drift analyses), or no governance /
reproducibility / monitoring metadata files were passed (see
`trustlens assess --help` for the `--reference`, `--metadata`,
`--governance-controls`, `--reproducibility-metadata`, and
`--monitoring-metadata` options).

### Limitations

Assessment results describe available evidence and indicators; they do not
independently certify trustworthiness, fairness, safety, or compliance.
Findings must be interpreted within your application, data context, and
applicable governance requirements.

## 6. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `trustlens: command not found` | The install directory is not on `PATH`; reinstall with `pip install trustlens-ai==1.0.0` or use `python -m trustlens.cli` |
| `No such file` for the CSV | Run from the directory containing the file, or pass an absolute path |
| Column errors / empty findings | Column names are case-sensitive — check they match the CSV header exactly |
| Missing drift/governance sections | Expected without `--reference` / metadata files; see `analysis_availability` |
| Different Python version warnings | Use Python 3.10–3.12 as tested in CI |

## 7. Next steps

- [Practical Examples](examples/README.md) — the unified Python example.
- [README](README.md) — full capability reference for each dimension.
- [CONTRIBUTING](CONTRIBUTING.md) — how to contribute back.
