
# Changelog

All notable changes to TrustLens AI are documented in this file.

TrustLens AI is an open-source Python framework for assessing data quality, data governance, AI readiness, responsible AI indicators, and operational trust.

This changelog follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses [Semantic Versioning](https://semver.org/) as its intended versioning approach.

Until the first stable release, the project remains under active development. Features, interfaces, and report structures may change.

## [Unreleased]

### Release Preparation

TrustLens AI is preparing for its first public stable release.

The initial stable release is planned as `1.0.0`, subject to final verification, documentation review, compatibility testing, and release approval.

The current development version is `0.1.0-dev`.

### Added

#### Unified Assessment Framework — Phase 7A

- Introduced `TrustLensAssessor` as the unified assessment interface.
- Integrated five core assessment dimensions:
  - Data Quality
  - Data Governance
  - AI Readiness
  - Responsible AI
  - Operational Trust
- Added structured unified assessment results.
- Added integration tests for the unified assessor.

#### Cross-Dimension Reporting — Phase 7B

- Introduced cross-dimension assessment reporting.
- Preserved detailed results from individual assessment components.
- Added a cross-dimension summary for consolidated interpretation.
- Added reporting integration tests.
- Maintained evidence-based interpretation without introducing an unsupported universal trust score.

#### Command-Line Interface — Phase 7C

- Introduced the `trustlens` command-line entry point.
- Added the `assess` command for CSV datasets.
- Added JSON report export.
- Added optional configuration for:
  - Target columns
  - Prediction columns
  - Sensitive attributes
  - Reference datasets
  - Governance metadata
  - Governance controls
  - Reproducibility metadata
  - Monitoring metadata
  - Identifier columns
- Added CLI help and input validation.
- Added protection against overwriting existing report files.
- Added CLI integration tests.

#### Packaging and Installation — Phase 7D

- Added Python package configuration using `pyproject.toml`.
- Configured setuptools as the package build backend.
- Added dynamic package version management.
- Registered the `trustlens` console command.
- Declared Python and package dependency requirements.
- Added development dependencies for testing and packaging.
- Added wheel and source distribution build validation.
- Added installed-package metadata verification.
- Added CLI installation checks.

#### Documentation and Examples — Phase 7E

- Expanded the main project README.
- Added installation and usage instructions.
- Documented the five assessment dimensions.
- Added unified assessment examples.
- Added a synthetic sample dataset.
- Added a practical examples guide.
- Documented CLI options and JSON report generation.
- Added automated validation of documentation examples.
- Expanded guidance on responsible interpretation and human oversight.

#### End-to-End Testing and CI — Phase 7F

- Added end-to-end CLI assessment tests.
- Added validation of five-dimension JSON reporting.
- Added tests for contextual assessment parameters.
- Added reference dataset and governance metadata scenarios.
- Added invalid-input tests.
- Added JSON compatibility checks.
- Added protection against accidental report overwriting.
- Expanded continuous integration verification.
- Updated GitHub Actions to Node.js 24-compatible action versions.

### Verified

The Phase 7 development branch has passed automated GitHub Actions checks for:

- Python 3.10
- Python 3.11
- Python 3.12

Continuous integration also verifies package builds, wheel installation, package metadata, CLI availability, automated tests, and JSON report generation.

Passing automated tests does not independently establish production suitability, regulatory compliance, or the absence of defects.

---

## Development History — Phases 1–6

The following sections summarise the major capabilities implemented before the unified assessment and release preparation work.

These sections represent development milestones rather than separately published package releases.

### Phase 6 — Operational Trust and Monitoring

#### Added

- Integrated `OperationalTrustAssessor`.
- Data drift assessment using current and reference datasets.
- Numeric mean-shift indicators.
- Categorical distribution-change indicators.
- Schema-change and feature-type compatibility indicators.
- Data stability assessment.
- Missingness and duplicate-row pressure indicators.
- Constant and near-constant feature indicators.
- High-cardinality feature indicators.
- Reproducibility readiness assessment.
- Dataset version and code version evidence checks.
- Execution environment and data lineage indicators.
- Monitoring readiness assessment.
- Monitoring ownership and review cadence indicators.
- Alerting, logging, and incident process evidence checks.
- Integrated Operational Trust reporting.

#### Design Principles

- Distinguishes available analyses from unavailable analyses.
- Identifies missing evidence requiring human review.
- Does not claim that supplied metadata proves reproducibility.
- Does not claim that monitoring configuration proves operational effectiveness.
- Avoids unsupported universal trust certification.

### Phase 5 — Responsible AI

#### Added

- Responsible AI assessment capabilities.
- Group fairness indicators.
- Bias indicator analysis.
- Prediction-performance fairness analysis.
- Explainability readiness indicators.
- Integrated Responsible AI reporting.

#### Design Principles

- Statistical differences are treated as assessment indicators.
- Findings require contextual interpretation.
- Absence of detected indicators does not establish fairness.
- Assessment results do not independently determine legal compliance.

### Phase 4 — AI Readiness

#### Added

- Integrated `AIReadinessAssessor`.
- Feature suitability analysis.
- Class imbalance indicators.
- Potential target leakage indicators.
- Structured AI readiness reporting.
- Assessment of dataset conditions relevant to machine-learning preparation.

#### Design Principles

- Readiness indicators identify conditions requiring investigation.
- Results do not guarantee model performance or deployment suitability.

### Phase 3 — Data Governance

#### Added

- Integrated `DataGovernanceAssessor`.
- Dataset column inventory.
- Sensitive-data indicators.
- Personally identifiable information indicators.
- Privacy-risk assessment.
- Metadata completeness analysis.
- Governance-control coverage analysis.
- Missing governance-control identification.
- Governance maturity indicators.
- Integrated governance reporting.

#### Foundational Governance Controls

- Data ownership
- Approved purpose
- Data classification
- Access control
- Retention policy
- Governance review process
- Accountability

#### Design Principles

- Governance scores and classifications are review indicators.
- Governance assessment does not establish legal or regulatory compliance.

### Phase 2 — Data Quality Enhancement

#### Added

- Enhanced completeness assessment.
- Uniqueness assessment.
- IQR-based outlier detection.
- Outlier summaries.
- Expanded structured data-quality reporting.

### Phase 1 — Data Quality Foundation

#### Added

- Initial TrustLens AI project structure.
- Foundational data-quality profiling.
- Dataset summary information.
- Missing-value analysis.
- Duplicate detection.
- Data-type analysis.
- Structured assessment output.
- Initial automated testing foundation.

---

## Release and Versioning Policy

TrustLens AI intends to follow Semantic Versioning after establishing its first stable public release.

Version numbers use the format:

`MAJOR.MINOR.PATCH`

- **MAJOR:** Changes that introduce incompatible public API behaviour.
- **MINOR:** Backward-compatible feature additions.
- **PATCH:** Backward-compatible bug fixes.

Pre-release identifiers such as `rc1` may be used for release candidates.

A release candidate is not a stable release and may still require corrections before publication.

### Planned Release Sequence

| Version | Purpose | Status |
|---|---|---|
| `0.1.0-dev` | Current development version | In use |
| `1.0.0rc1` | Initial release candidate | Planned |
| `1.0.0` | First stable public release | Planned |

Release versions will only be recorded as published releases after the corresponding release has been created.

## Responsible Use and Limitations

TrustLens AI provides assessment indicators to support research, development, governance, and informed decision-making.

It does not independently:

- Certify datasets or AI systems as trustworthy.
- Establish compliance with legislation or regulatory requirements.
- Prove the absence of bias or discrimination.
- Guarantee model performance.
- Verify the effectiveness of deployed monitoring systems.
- Replace independent validation or professional judgement.

Users remain responsible for assessing the suitability of the framework and its findings for their intended applications.

## Project Repository

Repository: https://github.com/deyanjukorede/trustlens-ai

For installation instructions, supported capabilities, examples, and current project status, see [README.md](README.md).
