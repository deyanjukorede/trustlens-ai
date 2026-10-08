
"""
Tests for TrustLens AI cross-dimension reporting.

These tests verify consolidated review indicators, unavailable
analyses, dimension coverage, input validation, and integration
with the unified assessment interface.
"""

import pandas as pd
import pytest

from trustlens.reporting import UnifiedReportBuilder
from trustlens.unified import TrustLensAssessor


@pytest.fixture
def sample_data():
    """Create a small dataset with modelling context."""
    return pd.DataFrame(
        {
            "age": [22, 34, 45, 29, 51, 38, 27, 43, 36, 49],
            "income": [
                25000, 42000, 58000, 31000, 67000,
                48000, 29000, 55000, 44000, 63000,
            ],
            "group": [
                "A", "B", "A", "B", "A",
                "B", "A", "B", "A", "B",
            ],
            "target": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
            "prediction": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_unified_report_contains_cross_dimension_summary(sample_data):
    """The unified assessor should include consolidated reporting."""
    report = TrustLensAssessor(sample_data).assess()

    assert "cross_dimension_summary" in report

    summary = report["cross_dimension_summary"]

    assert summary["report_type"] == "cross_dimension_summary"
    assert summary["dimension_count"] == 5
    assert len(summary["dimensions_assessed"]) == 5


def test_reporting_preserves_all_five_dimensions(sample_data):
    """Reporting must not remove specialist assessment results."""
    report = TrustLensAssessor(sample_data).assess()

    assert set(report["dimensions"]) == {
        "data_quality",
        "data_governance",
        "ai_readiness",
        "responsible_ai",
        "operational_trust",
    }

    assert "scores" in report["dimensions"]["data_quality"]
    assert "privacy_risk" in report["dimensions"]["data_governance"]
    assert "readiness_summary" in report["dimensions"]["ai_readiness"]

    assert (
        "responsible_ai_summary"
        in report["dimensions"]["responsible_ai"]
    )

    assert (
        "operational_trust_summary"
        in report["dimensions"]["operational_trust"]
    )


def test_reporting_identifies_missing_values(sample_data):
    """Missing values should produce a Data Quality review indicator."""
    data = sample_data.copy()
    data.loc[0, "income"] = None

    report = TrustLensAssessor(data).assess()

    indicators = report["cross_dimension_summary"]["review_indicators"]

    assert any(
        item["dimension"] == "data_quality"
        and item["indicator"] == "missing_values"
        for item in indicators
    )


def test_reporting_identifies_duplicate_rows(sample_data):
    """Duplicate rows should produce a Data Quality review indicator."""
    data = pd.concat(
        [sample_data, sample_data.iloc[[0]]],
        ignore_index=True,
    )

    report = TrustLensAssessor(data).assess()

    indicators = report["cross_dimension_summary"]["review_indicators"]

    assert any(
        item["dimension"] == "data_quality"
        and item["indicator"] == "duplicate_rows"
        for item in indicators
    )


def test_reporting_identifies_unavailable_governance_analysis(sample_data):
    """Missing governance evidence must be recorded as unavailable."""
    report = TrustLensAssessor(sample_data).assess()

    unavailable = report["cross_dimension_summary"][
        "unavailable_analyses"
    ]

    assert any(
        item["dimension"] == "data_governance"
        and item["analysis"] == "metadata_completeness"
        for item in unavailable
    )

    assert any(
        item["dimension"] == "data_governance"
        and item["analysis"] == "governance_controls"
        for item in unavailable
    )


def test_reporting_identifies_unavailable_readiness_analysis(sample_data):
    """Target-dependent readiness analyses require a target column."""
    report = TrustLensAssessor(sample_data).assess()

    unavailable = report["cross_dimension_summary"][
        "unavailable_analyses"
    ]

    assert any(
        item["dimension"] == "ai_readiness"
        and item["analysis"] == "class_imbalance"
        for item in unavailable
    )

    assert any(
        item["dimension"] == "ai_readiness"
        and item["analysis"] == "leakage_risk"
        for item in unavailable
    )


def test_reporting_identifies_unavailable_fairness_analysis(sample_data):
    """Fairness analysis without predictions must remain unavailable."""
    report = TrustLensAssessor(sample_data).assess()

    unavailable = report["cross_dimension_summary"][
        "unavailable_analyses"
    ]

    assert any(
        item["dimension"] == "responsible_ai"
        and item["analysis"] == "group_fairness"
        for item in unavailable
    )


def test_reporting_identifies_unavailable_drift_analysis(sample_data):
    """Drift analysis requires reference data."""
    report = TrustLensAssessor(sample_data).assess()

    unavailable = report["cross_dimension_summary"][
        "unavailable_analyses"
    ]

    assert any(
        item["dimension"] == "operational_trust"
        and item["analysis"] == "data_drift"
        for item in unavailable
    )


def test_reporting_with_reference_data_enables_drift(sample_data):
    """Reference data should remove drift from unavailable analyses."""
    report = TrustLensAssessor(
        sample_data,
        reference_data=sample_data.copy(),
    ).assess()

    unavailable = report["cross_dimension_summary"][
        "unavailable_analyses"
    ]

    assert not any(
        item["dimension"] == "operational_trust"
        and item["analysis"] == "data_drift"
        for item in unavailable
    )


def test_reporting_with_modelling_context_enables_fairness(sample_data):
    """Sufficient modelling inputs should enable fairness analysis."""
    report = TrustLensAssessor(
        sample_data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    ).assess()

    unavailable = report["cross_dimension_summary"][
        "unavailable_analyses"
    ]

    assert not any(
        item["dimension"] == "responsible_ai"
        and item["analysis"] == "group_fairness"
        for item in unavailable
    )

    assert not any(
        item["dimension"] == "responsible_ai"
        and item["analysis"] == "prediction_performance_fairness"
        for item in unavailable
    )


def test_reporting_counts_match_reported_items(sample_data):
    """Summary counts must match the corresponding collections."""
    report = TrustLensAssessor(sample_data).assess()

    summary = report["cross_dimension_summary"]

    assert summary["review_indicator_count"] == len(
        summary["review_indicators"]
    )

    assert summary["unavailable_analysis_count"] == len(
        summary["unavailable_analyses"]
    )

    assert summary["review_required"] == bool(
        summary["review_indicators"]
    )


def test_reporting_preserves_interpretation_limitations(sample_data):
    """Consolidated reporting must preserve assessment limitations."""
    report = TrustLensAssessor(sample_data).assess()

    summary = report["cross_dimension_summary"]

    assert summary["limitations"] == report["limitations"]
    assert isinstance(summary["interpretation"], str)
    assert len(summary["interpretation"]) > 0


def test_reporting_builder_can_be_used_independently(sample_data):
    """The reporting builder should accept a unified assessment."""
    assessment = TrustLensAssessor(sample_data).assess()

    summary = UnifiedReportBuilder(assessment).build()

    assert summary["report_type"] == "cross_dimension_summary"
    assert summary["dimension_count"] == 5


def test_reporting_builder_rejects_non_dictionary():
    """Reporting input must be a dictionary."""
    with pytest.raises(TypeError, match="assessment must be a dictionary"):
        UnifiedReportBuilder(None)


def test_reporting_builder_rejects_missing_dimensions():
    """All five dimensions must be present."""
    with pytest.raises(ValueError, match="missing dimensions"):
        UnifiedReportBuilder(
            {
                "dimensions": {
                    "data_quality": {},
                }
            }
        )


def test_reporting_builder_rejects_invalid_dimension_result():
    """Each dimension result must be a dictionary."""
    with pytest.raises(TypeError, match="must be a dictionary"):
        UnifiedReportBuilder(
            {
                "dimensions": {
                    "data_quality": None,
                    "data_governance": {},
                    "ai_readiness": {},
                    "responsible_ai": {},
                    "operational_trust": {},
                }
            }
        )


def test_reporting_preserves_original_assessment(sample_data):
    """Building the summary must not change the specialist results."""
    assessment = TrustLensAssessor(sample_data).assess()

    original_dimensions = assessment["dimensions"].copy()

    UnifiedReportBuilder(assessment).build()

    assert assessment["dimensions"] == original_dimensions


def test_reporting_with_no_quality_issues_does_not_invent_quality_review(
    sample_data,
):
    """Clean quality indicators should not create unsupported findings."""
    report = TrustLensAssessor(sample_data).assess()

    indicators = report["cross_dimension_summary"]["review_indicators"]

    assert not any(
        item["dimension"] == "data_quality"
        and item["indicator"] in {"missing_values", "duplicate_rows"}
        for item in indicators
    )
