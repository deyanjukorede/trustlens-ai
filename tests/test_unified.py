
"""
Integration tests for the TrustLens AI unified assessment interface.

These tests verify that the unified assessor coordinates the five
existing assessment dimensions, preserves specialist results,
reports analysis availability, and validates input data.
"""

import pandas as pd
import pytest

from trustlens.unified import TrustLensAssessor


@pytest.fixture
def sample_data():
    """Create a small dataset with modelling and fairness context."""
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


def test_unified_assessment_contains_all_five_dimensions(sample_data):
    """All five specialist assessments should appear in the report."""
    report = TrustLensAssessor(sample_data).assess()

    assert report["framework"] == "TrustLens AI"
    assert report["report_type"] == "unified_assessment"

    assert set(report["dimensions"]) == {
        "data_quality",
        "data_governance",
        "ai_readiness",
        "responsible_ai",
        "operational_trust",
    }

    for result in report["dimensions"].values():
        assert isinstance(result, dict)


def test_unified_assessment_preserves_dataset_context(sample_data):
    """Dataset dimensions and optional-context defaults are reported."""
    report = TrustLensAssessor(sample_data).assess()

    context = report["assessment_context"]

    assert context["dataset"]["rows"] == 10
    assert context["dataset"]["columns"] == 5
    assert context["target_column"] is None
    assert context["prediction_column"] is None
    assert context["sensitive_attributes"] == []
    assert context["reference_data_supplied"] is False


def test_unified_assessment_preserves_quality_results(sample_data):
    """The existing Data Quality report should remain accessible."""
    report = TrustLensAssessor(sample_data).assess()

    quality = report["dimensions"]["data_quality"]

    assert quality["dataset"]["rows"] == 10
    assert quality["dataset"]["columns"] == 5
    assert "missing_values" in quality
    assert "duplicates" in quality
    assert "outliers" in quality
    assert "scores" in quality
    assert "data_quality" in quality["scores"]


def test_unified_assessment_without_target_reports_unavailable_analysis(
    sample_data,
):
    """Target-dependent analyses must not be treated as completed."""
    report = TrustLensAssessor(sample_data).assess()

    readiness = report["analysis_availability"]["ai_readiness"]

    assert readiness["class_imbalance"] is False
    assert readiness["leakage_risk"] is False

    assert (
        report["dimensions"]["ai_readiness"]["class_imbalance"][
            "applicable"
        ]
        is False
    )


def test_unified_assessment_with_modelling_context(sample_data):
    """Target, predictions and sensitive attributes should propagate."""
    report = TrustLensAssessor(
        sample_data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    ).assess()

    context = report["assessment_context"]

    assert context["target_column"] == "target"
    assert context["prediction_column"] == "prediction"
    assert context["sensitive_attributes"] == ["group"]

    readiness = report["analysis_availability"]["ai_readiness"]

    assert readiness["class_imbalance"] is True
    assert readiness["leakage_risk"] is True

    responsible = report["dimensions"]["responsible_ai"]

    assert responsible["target_column"] == "target"
    assert responsible["prediction_column"] == "prediction"
    assert responsible["sensitive_attributes"] == ["group"]

    assert (
        responsible["analysis_availability"]["group_fairness"][
            "available"
        ]
        is True
    )

    assert (
        responsible["analysis_availability"][
            "prediction_performance_fairness"
        ]["available"]
        is True
    )


def test_unified_assessment_without_reference_data(sample_data):
    """Drift analysis should remain unavailable without a baseline."""
    report = TrustLensAssessor(sample_data).assess()

    operational = report["dimensions"]["operational_trust"]

    assert operational["data_drift"] is None

    assert (
        report["analysis_availability"]["operational_trust"][
            "data_drift"
        ]["available"]
        is False
    )


def test_unified_assessment_with_reference_data(sample_data):
    """Providing reference data should enable drift assessment."""
    reference = sample_data.copy()

    report = TrustLensAssessor(
        sample_data,
        reference_data=reference,
    ).assess()

    operational = report["dimensions"]["operational_trust"]

    assert report["assessment_context"]["reference_data_supplied"] is True
    assert operational["data_drift"] is not None

    assert (
        report["analysis_availability"]["operational_trust"][
            "data_drift"
        ]["available"]
        is True
    )


def test_unified_assessment_without_governance_context(sample_data):
    """Missing optional governance evidence must remain unavailable."""
    report = TrustLensAssessor(sample_data).assess()

    governance = report["dimensions"]["data_governance"]

    assert governance["metadata_completeness"] is None
    assert governance["governance_controls"] is None

    assert (
        report["analysis_availability"]["governance"][
            "metadata_completeness"
        ]
        is False
    )

    assert (
        report["analysis_availability"]["governance"][
            "governance_controls"
        ]
        is False
    )


def test_unified_assessment_preserves_limitations(sample_data):
    """The unified report must include interpretation limitations."""
    report = TrustLensAssessor(sample_data).assess()

    assert isinstance(report["limitations"], list)
    assert len(report["limitations"]) >= 1
    assert all(
        isinstance(limitation, str)
        for limitation in report["limitations"]
    )


def test_analyze_alias_returns_unified_report(sample_data):
    """The analyze alias should produce the unified report."""
    report = TrustLensAssessor(sample_data).analyze()

    assert report["report_type"] == "unified_assessment"
    assert len(report["dimensions"]) == 5


@pytest.mark.parametrize(
    "invalid_data",
    [
        None,
        [],
        {"age": [22, 34]},
        "not a dataframe",
    ],
)
def test_unified_assessment_rejects_non_dataframe(invalid_data):
    """Non-DataFrame inputs should raise TypeError."""
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        TrustLensAssessor(invalid_data)


def test_unified_assessment_rejects_empty_dataframe():
    """An empty DataFrame should raise ValueError."""
    with pytest.raises(ValueError, match="data must not be empty"):
        TrustLensAssessor(pd.DataFrame())


def test_unified_assessment_rejects_invalid_sensitive_attributes(
    sample_data,
):
    """Sensitive attributes must be provided as a list or None."""
    with pytest.raises(TypeError, match="sensitive_attributes"):
        TrustLensAssessor(
            sample_data,
            sensitive_attributes="group",
        )


def test_unified_assessment_rejects_unknown_target(sample_data):
    """Unknown target columns should be rejected by the specialist."""
    with pytest.raises(ValueError, match="target_column"):
        TrustLensAssessor(
            sample_data,
            target_column="unknown_target",
        ).assess()


def test_unified_assessment_rejects_unknown_prediction(sample_data):
    """Unknown prediction columns should be rejected."""
    with pytest.raises(ValueError, match="prediction_column"):
        TrustLensAssessor(
            sample_data,
            prediction_column="unknown_prediction",
        ).assess()


def test_unified_assessment_rejects_unknown_sensitive_attribute(
    sample_data,
):
    """Unknown sensitive attributes should be rejected."""
    with pytest.raises(ValueError, match="sensitive attributes"):
        TrustLensAssessor(
            sample_data,
            sensitive_attributes=["unknown_group"],
        ).assess()
