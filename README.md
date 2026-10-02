# TrustLens AI

### Open-Source AI Data Readiness & Trust Assessment Framework

TrustLens AI is an open-source framework designed to evaluate whether data is ready, reliable, responsibly governed, and trustworthy enough for use in artificial intelligence and machine learning systems.

Rather than assessing data quality alone, TrustLens AI is being developed as a unified assessment framework across five core dimensions:

1. **Data Quality** — completeness, consistency, validity, duplicates, and anomalies.
2. **Data Governance** — privacy, sensitive data, metadata, ownership, and governance controls.
3. **AI Readiness** — class imbalance, feature suitability, leakage risks, and modelling readiness.
4. **Responsible AI** — fairness, explainability, and potential bias.
5. **Operational Trust** — data drift, reproducibility, monitoring, and ongoing reliability.

## Project Vision

The goal of TrustLens AI is to bridge the gap between traditional data-quality assessment, data governance, and responsible AI.

As the framework develops, TrustLens AI is intended to generate two broader indicators:

### Data Readiness Score

A quantitative assessment of whether a dataset is sufficiently prepared for machine learning and AI applications.

### AI Trust Score

A broader assessment incorporating data quality, governance, privacy, fairness, explainability, and operational considerations.

## Current Capabilities

TrustLens AI currently provides foundational capabilities across data quality and data governance.

### Data Quality

Current data-quality functionality includes:

- Dataset profiling
- Missing-value analysis
- Duplicate detection
- Data-type analysis
- Completeness assessment
- Uniqueness assessment
- IQR-based outlier detection
- Outlier summaries and affected-column reporting
- Structured data-quality assessment output

### Data Governance

The Phase 3 governance engine introduces:

- Dataset governance assessment
- Column inventory generation
- Sensitive-data detection
- Detection of sensitive information from column names and value patterns
- Personally identifiable information indicators
- Privacy-risk assessment
- Sensitive-column ratio assessment
- Privacy exposure scoring
- Privacy-risk classification
- Governance recommendations
- Metadata completeness assessment
- Metadata coverage scoring
- Identification of missing metadata
- Governance-control assessment
- Governance-control coverage scoring
- Identification of missing governance controls
- Governance maturity classification
- Integrated governance reporting

The governance capabilities are exposed through the `DataGovernanceAssessor`, allowing structural assessment, sensitive-data analysis, privacy-risk assessment, metadata completeness, and governance controls to be evaluated through a unified interface.

## Governance Controls

TrustLens AI currently evaluates seven foundational governance controls:

1. Data ownership
2. Approved purpose
3. Data classification
4. Access control
5. Retention policy
6. Governance review process
7. Accountability

The framework can determine which controls are implemented or missing, calculate governance-control coverage, classify governance maturity, and generate recommendations for missing controls.

## Phase 3 Integrated Governance Report

When governance information is supplied, the integrated assessment can expose the following sections:

- `dataset`
- `column_inventory`
- `sensitive_data`
- `privacy_risk`
- `metadata_completeness`
- `governance_controls`

Optional governance inputs remain backward compatible, allowing datasets to be assessed even when metadata or governance-control information is not supplied.

## Planned Capabilities

Future development is expected to extend TrustLens AI with:

- Feature suitability analysis
- Class imbalance analysis
- Potential data-leakage detection
- AI readiness scoring
- Bias and fairness assessment
- Model explainability support
- Data-drift detection
- Reproducibility assessment
- Operational monitoring indicators
- Automated remediation recommendations
- Data Readiness Score
- AI Trust Score
- Interactive assessment dashboard
- Automated reports
- REST API

## Initial Use Cases

TrustLens AI is being designed for use across data-intensive environments, including:

- Financial services and fintech
- Fraud detection
- Credit-risk modelling
- Healthcare analytics
- Enterprise information systems
- Responsible AI governance

The framework is intended to support data scientists, analysts, governance professionals, researchers, AI developers, and organisations seeking to understand whether their data is sufficiently reliable and governed for AI use.

## Project Status

**Current Version:** Pre-Alpha

TrustLens AI is under active development.

The foundational Data Quality capabilities and Phase 3 Data Governance capabilities are now implemented. Development will continue toward AI readiness, responsible AI, operational trust, scoring, reporting, and deployment capabilities.

The current implementation includes automated tests across supported Python environments, with GitHub Actions used for continuous integration.

## Technology Stack

The project currently uses or is expected to use:

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- SHAP
- Pytest
- Docker
- GitHub Actions

## Development Roadmap

### Phase 1: Data Quality Foundation — Implemented

Core dataset profiling, missing-value analysis, duplicate detection, completeness assessment, uniqueness assessment, and foundational data-quality reporting.

### Phase 2: Data Quality Enhancement — Implemented

IQR-based outlier detection and integration of anomaly information into the data-quality assessment workflow.

### Phase 3: Data Governance — Implemented

Sensitive-data detection, privacy-risk assessment, metadata completeness assessment, governance controls, governance scoring, recommendations, and integrated governance reporting.

### Phase 4: AI Readiness — Planned

Feature suitability, class imbalance, leakage-risk analysis, and AI readiness assessment.

### Phase 5: Responsible AI — Planned

Bias detection, fairness metrics, explainability, and responsible-AI assessment.

### Phase 6: Operational Trust — Planned

Data drift, reproducibility, monitoring, and ongoing reliability indicators.

### Phase 7: Scoring, Reporting & Platform — Planned

Data Readiness Score, AI Trust Score, automated reporting, interactive assessment dashboard, REST API, and deployment capabilities.

## Testing

TrustLens AI uses `pytest` for automated testing and GitHub Actions for continuous integration.

The test suite currently covers data-quality and governance components, including integration tests designed to verify that Phase 3 governance capabilities operate together without breaking existing assessment behaviour.

## Contributing

TrustLens AI is currently in early development. Contribution guidelines will be introduced as the framework matures.

Contributions, testing feedback, research collaboration, and suggestions for improving the framework are welcome as the project develops.

## License

This project is licensed under the MIT License.
