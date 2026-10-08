
# Contributing to TrustLens AI

Thank you for your interest in contributing to TrustLens AI.

TrustLens AI is an open-source Python framework for assessing data quality, data governance, AI readiness, responsible AI indicators, and operational trust.

We welcome contributions from software developers, data scientists, researchers, AI governance professionals, cybersecurity specialists, students, and organisations interested in trustworthy AI and responsible data practices.

Our goal is to build a transparent, extensible, reliable, and evidence-based framework that supports informed decisions throughout the data and AI lifecycle.

## 1. Guiding Principles

Contributions to TrustLens AI should support the following principles:

- **Transparency:** Assessment methods, assumptions, and limitations should be clearly documented.
- **Evidence-based evaluation:** Findings should be supported by measurable indicators or explicitly supplied evidence.
- **Responsible AI:** Fairness, privacy, governance, and safety implications should be considered.
- **Human oversight:** Automated assessments should support, not replace, informed human judgement.
- **Reproducibility:** Analyses should be testable and reproducible wherever practical.
- **Security and privacy:** Changes should minimise unnecessary data exposure and security risks.
- **Reliability:** New functionality should include appropriate validation and automated tests.
- **Extensibility:** Contributions should preserve a maintainable and modular architecture.

TrustLens AI must not present assessment indicators as independent certifications of fairness, trustworthiness, regulatory compliance, or production safety.

## 2. Ways to Contribute

Contributions may include:

- Bug reports and reproducible defect investigations
- Improvements to existing assessment methods
- New data-quality indicators
- Data governance and privacy assessment enhancements
- AI readiness and machine-learning evaluation improvements
- Responsible AI and fairness analysis
- Operational trust and monitoring improvements
- Security and reliability enhancements
- Performance optimisation
- Automated tests and validation datasets
- Documentation and tutorials
- Accessibility and developer-experience improvements
- Research proposals and methodological reviews

For substantial new features, please open a GitHub issue to discuss the proposal before beginning implementation.

This helps avoid duplicated work and ensures proposed changes align with the project's architecture and objectives.

## 3. Development Environment

### Prerequisites

TrustLens AI currently tests against:

- Python 3.10
- Python 3.11
- Python 3.12

Git and a suitable Python development environment are recommended.

### Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/trustlens-ai.git
cd trustlens-ai
```

Replace `YOUR_USERNAME` with your GitHub username.

### Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Development Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs TrustLens AI in editable mode together with its declared development dependencies.

## 4. Development Workflow

### Create a Feature Branch

Create a branch for your proposed change:

```bash
git checkout -b feature/your-feature-name
```

Suggested branch naming conventions:

- `feature/` for new functionality
- `fix/` for bug fixes
- `docs/` for documentation
- `test/` for automated tests
- `refactor/` for internal code improvements
- `ci/` for continuous integration changes

Avoid making unrelated changes in the same pull request.

### Implement Your Changes

Keep changes focused and consistent with the existing project architecture.

Where practical:

- Reuse existing interfaces and conventions.
- Avoid unnecessary dependencies.
- Provide clear validation and error messages.
- Preserve compatibility with supported Python versions.
- Avoid silently changing established report structures.
- Document methodological assumptions and limitations.
- Consider privacy and security implications.

### Run Automated Tests

From the repository root:

```bash
python -m pytest -v
```

All existing tests should pass before a pull request is submitted.

If a change introduces new functionality, include tests demonstrating its intended behaviour.

If a change fixes a bug, include a regression test where practical.

### Validate Package Installation

For changes affecting packaging or the CLI:

```bash
python -m build
```

To install the package locally:

```bash
python -m pip install -e .
```

To verify CLI availability:

```bash
trustlens --help
trustlens assess --help
```

The build command requires the `build` development dependency.

## 5. Testing Expectations

Contributions should include testing appropriate to their scope.

Relevant test categories include:

- Unit tests
- Input validation tests
- Edge-case tests
- Integration tests
- End-to-end tests
- Regression tests
- CLI tests
- JSON reporting tests
- Documentation example tests

Tests should use synthetic, anonymised, or appropriately authorised data.

Do not commit confidential organisational data, credentials, access tokens, financial account details, or personal information.

When introducing an assessment method, test both expected and exceptional conditions.

Tests should verify the method's actual contract rather than assume that a particular score or indicator proves real-world trustworthiness.

## 6. Assessment Methodology Standards

TrustLens AI is intended to support evidence-based evaluation.

New assessment methods should document:

1. The purpose of the assessment
2. Required input data
3. Underlying assumptions
4. Calculation or evaluation methodology
5. Configurable thresholds, where applicable
6. Expected output structure
7. Limitations and potential sources of error
8. Conditions requiring human review
9. Relevant references or supporting research, where appropriate

Avoid introducing unexplained scores, arbitrary classifications, or unsupported claims.

Where a score is proposed, contributors should explain its interpretation and limitations and provide appropriate validation evidence.

A numerical indicator must not automatically be described as a certification of trustworthiness, fairness, safety, or legal compliance.

## 7. Responsible AI and Fairness Contributions

Contributions involving fairness, bias, explainability, or model evaluation require particular care.

Contributors should:

- Identify the population and context to which an analysis applies.
- Document the assumptions underlying statistical measures.
- Explain when an analysis is unavailable or insufficiently supported.
- Distinguish statistical indicators from legal or ethical conclusions.
- Consider limitations arising from small samples and incomplete data.
- Avoid implying that the absence of a detected issue proves fairness.
- Explain where additional domain expertise or human review is required.

Assessment outputs should communicate uncertainty and evidence gaps where appropriate.

## 8. Data Governance, Privacy, and Security

Contributors must not introduce functionality that unnecessarily exposes sensitive information.

When developing data-processing features:

- Minimise unnecessary retention of source data.
- Avoid logging sensitive values by default.
- Validate external inputs.
- Use secure dependency and file-handling practices.
- Avoid embedding credentials or secrets in source code.
- Document relevant privacy limitations.
- Use synthetic datasets in public examples wherever possible.

If you discover a potentially exploitable security vulnerability, do not publish technical exploitation details in a public issue.

Please follow the project's `SECURITY.md` reporting instructions once available. Until then, use GitHub's private vulnerability reporting feature if enabled, or contact the repository maintainer privately through an established contact channel.

## 9. Documentation Standards

Documentation contributions should be:

- Accurate
- Clear
- Reproducible
- Accessible to the intended audience
- Consistent with implemented functionality

Where appropriate, new features should include:

- Installation or configuration instructions
- Usage examples
- Input requirements
- Expected output descriptions
- Known limitations
- Testing instructions

Do not document planned features as though they are already implemented.

For significant user-facing changes, update the relevant README, examples, and changelog.

## 10. Code Quality and Maintainability

Contributors are encouraged to:

- Follow established Python conventions.
- Use descriptive names.
- Keep functions and classes focused.
- Add docstrings to public interfaces.
- Prefer clear implementation over unnecessary complexity.
- Handle invalid inputs explicitly.
- Avoid unnecessary changes to public APIs.
- Consider performance implications for larger datasets.
- Keep third-party dependencies to a reasonable minimum.

Compatibility-sensitive changes should be clearly identified in the pull request.

## 11. Commit Message Guidelines

Use descriptive commit messages that explain the purpose of each change.

Recommended prefixes include:

- `feat:` — new functionality
- `fix:` — bug correction
- `docs:` — documentation
- `test:` — automated tests
- `refactor:` — internal restructuring
- `ci:` — continuous integration
- `build:` — packaging or build changes
- `chore:` — maintenance

Examples:

```text
feat: add configurable dataset validation
fix: handle missing reference columns
docs: expand unified assessment examples
test: add regression tests for JSON export
ci: improve Python compatibility checks
```

## 12. Pull Request Guidelines

When submitting a pull request:

1. Describe the problem or improvement.
2. Explain the proposed implementation.
3. Reference relevant GitHub issues, where applicable.
4. Identify affected assessment components.
5. Summarise the tests performed.
6. Describe any changes to public interfaces or report structures.
7. Update documentation where needed.
8. Disclose known limitations or unresolved concerns.
9. Confirm that no sensitive information has been committed.

Pull requests may be reviewed for:

- Technical correctness
- Test coverage
- Security and privacy implications
- Methodological transparency
- Documentation quality
- Compatibility
- Maintainability
- Alignment with the project's objectives

Maintainers may request changes before accepting a contribution.

Submission of a pull request does not guarantee acceptance.

## 13. Continuous Integration

TrustLens AI uses GitHub Actions to validate supported Python environments.

The current CI workflow includes:

- Python 3.10 testing
- Python 3.11 testing
- Python 3.12 testing
- Automated pytest execution
- Package build validation
- Wheel installation
- Package metadata checks
- CLI availability checks
- CLI assessment and JSON report validation

Contributors should investigate and resolve failures introduced by their changes.

Passing CI checks does not replace appropriate code review, methodological evaluation, or security assessment.

## 14. Dependencies and Third-Party Code

Before adding a dependency:

- Explain why it is required.
- Consider whether existing dependencies can provide the functionality.
- Review licensing compatibility.
- Consider maintenance and security implications.
- Avoid unnecessary dependency expansion.

Do not submit copied proprietary code or material without appropriate rights and attribution.

Contributors are responsible for ensuring that their submissions may lawfully be distributed under the project's licence.

## 15. Research and Academic Contributions

Research-driven contributions are welcome.

Where appropriate:

- Cite relevant academic or technical sources.
- Describe the methodology sufficiently for independent review.
- Identify datasets and evaluation conditions.
- Explain methodological limitations.
- Avoid overstating experimental findings.
- Distinguish experimental features from established functionality.

Research contributions should aim to improve the framework's transparency, reproducibility, reliability, and practical usefulness.

## 16. Community Conduct

Contributors are expected to communicate respectfully and professionally.

Disagreement over technical decisions should focus on evidence, methodology, and implementation rather than personal criticism.

The project's `CODE_OF_CONDUCT.md`, once added, will provide further guidance on expected community behaviour.

## 17. Contribution Review and Acceptance

Project maintainers are responsible for reviewing and deciding whether proposed contributions should be merged.

Review decisions may consider:

- Relevance to the project roadmap
- Technical and methodological quality
- Security and privacy
- Test coverage
- Maintenance burden
- Compatibility with existing features
- Availability of review resources

Maintainers may request additional documentation, testing, or clarification before accepting changes.

## 18. Project Resources

- [Repository](https://github.com/deyanjukorede/trustlens-ai)
- [Project README](README.md)
- [Development Changelog](CHANGELOG.md)
- [Practical Examples](examples/README.md)
- [MIT License](LICENSE)

Thank you for helping build TrustLens AI into a more reliable, transparent, and useful open-source framework.
