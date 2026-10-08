# TrustLens AI

### Open-Source AI Data Readiness, Governance & Trust Assessment Framework

TrustLens AI is an open-source Python framework for evaluating data quality, data governance, AI readiness, responsible AI indicators, and operational trust.

Rather than focusing solely on data quality, TrustLens AI brings together five complementary assessment dimensions to help researchers, developers, data scientists, governance professionals, and organisations identify potential weaknesses in data and AI workflows.

The framework is designed to provide transparent, evidence-based indicators that support informed decisions and human review.

**Important:** TrustLens AI does not independently certify that a dataset or AI system is trustworthy, fair, reliable, legally compliant, or suitable for production deployment.

## Project Vision

TrustLens AI aims to bridge the gap between traditional data-quality assessment, data governance, machine-learning readiness, responsible AI, and ongoing operational monitoring.

The long-term vision is to establish an extensible framework that helps organisations evaluate data and AI systems throughout their lifecycle.

The project prioritises:

- Transparent assessment methods
- Reproducible analysis
- Explainable assessment results
- Responsible interpretation of findings
- Evidence-based governance
- Human oversight
- Extensible architecture
- Automated testing and continuous integration

## Core Assessment Dimensions

| Dimension | Focus |
|---|---|
| Data Quality | Completeness, consistency, validity, uniqueness, duplicates, and anomalies |
| Data Governance | Privacy, sensitive data, metadata, ownership, and governance controls |
| AI Readiness | Class imbalance, feature suitability, leakage risks, and modelling readiness |
| Responsible AI | Group fairness, bias indicators, prediction-performance fairness, and explainability readiness |
| Operational Trust | Data drift, stability, reproducibility readiness, monitoring readiness, and operational review indicators |

## Current Capabilities

### 1. Data Quality

The Data Quality component provides foundational dataset analysis, including:

- Dataset profiling
- Missing-value analysis
- Duplicate detection
- Data-type analysis
- Completeness assessment
- Uniqueness assessment
- IQR-based outlier detection
- Outlier summaries
- Structured assessment output

These capabilities help identify potential data-quality problems that may affect downstream analysis or modelling.

### 2. Data Governance

The Data Governance component supports:

- Dataset governance assessment
- Column inventory generation
- Sensitive-data detection
- Sensitive-information indicators based on column names and value patterns
- Personally identifiable information indicators
- Privacy-risk assessment
- Sensitive-column ratio analysis
- Privacy exposure scoring
- Privacy-risk classification
- Governance recommendations
- Metadata completeness assessment
- Metadata coverage scoring
- Missing metadata identification
- Governance-control assessment
- Governance-control coverage scoring
- Missing governance-control identification
- Governance maturity classification
- Integrated governance reporting

The integrated governance capabilities are exposed through `DataGovernanceAssessor`.

#### Foundational Governance Controls

TrustLens AI evaluates seven foundational governance controls:

1. Data ownership
2. Approved purpose
3. Data classification
4. Access control
5. Retention policy
6. Governance review process
7. Accountability

Governance coverage and maturity classifications are assessment indicators, not legal compliance certifications.

### 3. AI Readiness

The AI Readiness component provides indicators to support the preparation of datasets for machine-learning applications.

Implemented capabilities include:

- Feature suitability analysis
- Class imbalance analysis
- Potential data-leakage indicators
- Integrated AI readiness assessment

These analyses help identify conditions that may require additional investigation before model development.

AI readiness indicators do not guarantee model performance or suitability for a particular application.

### 4. Responsible AI

The Responsible AI component supports evidence-based analysis across several areas:

- Group fairness indicators
- Bias indicator analysis
- Prediction-performance fairness
- Explainability readiness
- Integrated Responsible AI assessment

The framework is designed to identify measurable differences, evidence gaps, and conditions requiring human review.

**Responsible AI assessment limitations:**

- A detected statistical difference does not automatically establish unlawful discrimination.
- Absence of a detected indicator does not prove that a model is fair.
- Explainability readiness does not establish that an explanation is correct or sufficient.
- Responsible AI findings require appropriate domain expertise, contextual interpretation, and human oversight.

### 5. Operational Trust

The Operational Trust component provides an integrated assessment of four operational dimensions.

#### Data Drift Detection

The `DataDriftAnalyzer` supports:

- Comparison of current and reference datasets
- Shared-column identification
- Schema-change indicators
- Numeric mean-shift analysis
- Categorical distribution-change analysis
- Feature-type incompatibility indicators
- Configurable drift thresholds
- Structured review indicators

Data drift analysis requires a reference dataset.

#### Data Stability and Change Indicators

The `DataStabilityAnalyzer` evaluates:

- Missingness pressure
- Constant features
- Near-constant features
- High-cardinality categorical features
- Duplicate-row pressure
- Feature-type summaries
- Configurable thresholds
- Stability review reasons

These indicators help identify structural conditions that may warrant further investigation.

#### Reproducibility Readiness

The `ReproducibilityReadinessAnalyzer` assesses supplied reproducibility evidence, including:

- Dataset version
- Code version
- Random seed
- Execution environment
- Data source
- Data lineage
- Execution identifier
- Optional identifier-column integrity

Missing evidence is reported as a review indicator.

The presence of metadata does not independently demonstrate that an experiment or pipeline is reproducible.

#### Monitoring Readiness

The `MonitoringReadinessAnalyzer` evaluates supplied monitoring evidence, including:

- Monitoring metrics
- Alerting rules
- Monitoring ownership
- Review cadence
- Logging configuration
- Incident processes
- Reassessment triggers
- Monitoring history

The framework identifies missing monitoring evidence and configuration conditions that require review.

Monitoring readiness assessment does not verify that a live monitoring system is operating effectively.

#### Integrated Operational Trust Assessment

The `OperationalTrustAssessor` coordinates all four operational dimensions.

It produces:

- Dataset assessment context
- Analysis availability information
- Data drift results, when reference data is available
- Data stability results
- Reproducibility readiness results
- Monitoring readiness results
- An integrated Operational Trust overview
- An Operational Trust summary

The integrated overview distinguishes between available and unavailable analyses and identifies which assessed dimensions contain review indicators.

It deliberately avoids reducing these findings to an unsupported numerical trust score.

## Getting Started

### Requirements

- Python 3.10, 3.11, or 3.12 (the versions tested in GitHub Actions)
- A CSV dataset for command-line assessment

### Installation from Source

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/deyanjukorede/trustlens-ai.git
cd trustlens-ai
python -m pip install -e .
```

Until the Phase 7 work is merged, use the `feature/unified-reporting-v1` branch to try these pre-release features:

```bash
git checkout feature/unified-reporting-v1
python -m pip install -e .
```

No stable package release has been published yet. These instructions install from source rather than from PyPI.

### First Assessment Using the CLI

Create a file named `sample.csv` with the following contents:

```csv
age,income,group,target
22,25000,A,0
34,42000,B,1
45,58000,A,1
29,31000,B,0
51,67000,A,1
38,48000,B,0
27,29000,A,0
43,55000,B,1
36,44000,A,0
49,63000,B,1
```

From the repository root, run:

```bash
trustlens assess sample.csv --output report.json
```

The command produces a JSON report at `report.json`. The CLI requires an output path and will not overwrite an existing file. Choose a new filename for subsequent runs.

For CLI help:

```bash
trustlens --help
trustlens assess --help
```

You can also run the module directly:

```bash
python -m trustlens.cli assess sample.csv --output report-2.json
```

### Command-Line Options

| Option | Purpose |
|---|---|
| `dataset` | Required positional path to a non-empty CSV dataset |
| `--output` | Required destination for a new JSON report |
| `--target` | Optional prediction target column |
| `--prediction` | Optional model prediction column |
| `--sensitive` | One or more sensitive-attribute column names |
| `--reference` | Reference CSV for data drift comparison |
| `--metadata` | Governance metadata JSON object file |
| `--governance-controls` | Governance controls JSON object file |
| `--reproducibility-metadata` | Reproducibility metadata JSON object file |
| `--monitoring-metadata` | Monitoring metadata JSON object file |
| `--identifier` | Optional identifier column |

For example, using a target and sensitive attribute:

```bash
trustlens assess sample.csv --target target --sensitive group --output targeted-report.json
```

To compare a dataset with a reference CSV:

```bash
trustlens assess sample.csv --reference reference.csv --output drift-report.json
```

`reference.csv` must exist and contain data before this command can run. Optional metadata files must contain JSON objects, not top-level JSON arrays.

### Unified Python Interface

TrustLens AI also provides `TrustLensAssessor` for developers who want structured assessment results in Python:

```python
import pandas as pd
from trustlens.unified import TrustLensAssessor

data = pd.read_csv("sample.csv")
assessor = TrustLensAssessor(data=data)
report = assessor.assess()

print(report.keys())
```

The unified interface coordinates Data Quality, Data Governance, AI Readiness, Responsible AI, and Operational Trust. Optional context can enable analyses that require additional inputs. The report preserves specialist assessment results and provides cross-dimension reporting rather than an unsupported overall trust certification.

### Interpreting Reports

- Review each dimension's detailed results and any reported evidence gaps.
- An unavailable or not-applicable analysis is not evidence that no risk exists.
- A review indicator is a prompt for further investigation, not a legal or safety determination.
- The output is intended to support human oversight, domain-specific validation, and appropriate governance decisions.
- Consider whether the dataset contains personal or confidential information before sharing assessment reports.

## Operational Trust Usage Example

The following example illustrates how to run an Operational Trust assessment using Pandas.

```python
import pandas as pd

from trustlens.operational_trust.assessor import OperationalTrustAssessor

reference_data = pd.DataFrame(
    {
        "record_id": [1, 2, 3, 4],
        "score": [10, 20, 30, 40],
    }
)

current_data = pd.DataFrame(
    {
        "record_id": [1, 2, 3, 4],
        "score": [12, 22, 32, 42],
    }
)

reproducibility_metadata = {
    "dataset_version": "v1.0",
    "code_version": "commit-abc123",
    "random_seed": 42,
    "environment": "python-3.12",
    "data_source": "validated-source",
    "lineage": "source-to-model-pipeline",
    "execution_id": "run-001",
}

monitoring_metadata = {
    "monitoring_metrics": ["data_drift", "missingness"],
    "alerting_rules": {"data_drift": "review_threshold"},
    "monitoring_owner": "ml-platform",
    "review_cadence": "monthly",
    "logging_enabled": True,
    "incident_process": "operational-runbook",
    "reassessment_triggers": ["material_data_change"],
    "monitoring_history": ["2026-10-review"],
}

assessor = OperationalTrustAssessor(
    current_data,
    reference_data=reference_data,
    reproducibility_metadata=reproducibility_metadata,
    identifier_column="record_id",
    monitoring_metadata=monitoring_metadata,
)

report = assessor.assess()

print(report["operational_trust_summary"])
print(report["operational_trust_overview"])
```

The returned report contains structured dictionaries that can be inspected programmatically or incorporated into downstream reporting workflows.

## Development Installation and Testing

For contributors working from a repository checkout:

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -v
```

Run these commands from the repository root. The project's `pyproject.toml` defines its Python package and CLI entry point. The legacy `requirements.txt` remains in the repository for existing workflows.

## Testing and Continuous Integration

TrustLens AI uses `pytest` for automated testing and GitHub Actions for continuous integration.

The test suite covers individual analyzers, assessment contracts, validation behaviour, and integration between components.

The CI workflow runs tests against:

- Python 3.10
- Python 3.11
- Python 3.12

The packaging validation workflow also builds wheel and source distributions, installs the built wheel, checks package metadata and the CLI entry point, and runs a CLI-to-JSON integration scenario. These automated checks help detect regressions as new functionality is introduced.

Passing tests provide evidence that tested behaviours work as expected; they do not establish that the framework is free from defects.

## Initial Use Cases

TrustLens AI is being developed for data-intensive environments, including:

- Financial services and fintech
- Fraud detection
- Credit-risk modelling
- Healthcare analytics
- Enterprise information systems
- Responsible AI governance
- Academic research
- Data governance and risk management

Assessment findings must be interpreted in the context of the relevant dataset, application, organisational controls, and regulatory environment.

## Development Roadmap

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Data Quality Foundation | Implemented |
| Phase 2 | Data Quality Enhancement | Implemented |
| Phase 3 | Data Governance | Implemented |
| Phase 4 | AI Readiness | Implemented |
| Phase 5 | Responsible AI | Implemented |
| Phase 6 | Operational Trust & Monitoring | Implemented |
| Phase 7 | Unified Reporting, Developer Experience & v1.0 Release | In progress |

### Phase 7 — Release Preparation Progress

| Workstream | Focus | Status |
|---|---|---|
| 7A | Unified Assessment Foundation | Implemented and tested |
| 7B | Cross-Dimension Reporting | Implemented and tested |
| 7C | Command-Line Interface and Developer Experience | Implemented and tested |
| 7D | Packaging and Installation | Implemented; CI checks passing |
| 7E | Documentation and Examples | In progress |
| 7F | End-to-End Testing and CI | Planned |
| 7G | v1.0 Release Preparation | Planned |

The Phase 7 changes are under development on the `feature/unified-reporting-v1` branch and are not yet a published stable release.

Longer-term opportunities may include:

- Interactive assessment dashboard
- Automated report generation
- REST API
- Deployment integrations
- Context-sensitive remediation recommendations
- Carefully validated assessment scoring methodologies

Any future numerical scoring methodology should document its assumptions, limitations, weighting, validation, and intended interpretation.

## Project Status

**Current status: Pre-release — Active Development**

TrustLens AI has implemented foundational assessment capabilities across its five core dimensions.

The unified assessment interface, cross-dimension reporting, CLI, and packaging workflow have been implemented on the Phase 7 development branch. Documentation, broader end-to-end validation, and release preparation remain in progress.

The framework is intended to support evidence-based evaluation rather than replace professional judgement, independent validation, or regulatory assessment.

## Technology Stack

The current implementation primarily uses:

- Python
- Pandas
- NumPy
- Pytest
- GitHub Actions

Additional technologies may be introduced as the project develops, including Scikit-learn, FastAPI, Streamlit, SHAP, and Docker, where appropriate to the implementation roadmap.

## Contributing

TrustLens AI is an open-source project under active development.

Contributions, testing feedback, research collaboration, documentation improvements, and suggestions for extending the assessment framework are welcome.

Formal contribution guidelines will be introduced as the project matures.

## License

TrustLens AI is licensed under the MIT License. See the repository's `LICENSE` file for details.
