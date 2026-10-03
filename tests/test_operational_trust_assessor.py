import pandas as pd
import pytest

from trustlens.operational_trust.assessor import OperationalTrustAssessor


def test_assessor_initialises_with_valid_dataframe():
    """Assessor should initialise with valid current data."""
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": ["A", "B", "C"],
        }
    )

    assessor = OperationalTrustAssessor(data)

    assert assessor.row_count == 3
    assert assessor.column_count == 2
    assert assessor.reference_data_available is False


def test_assessor_rejects_non_dataframe_input():
    """Current data must be supplied as a pandas DataFrame."""
    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
        OperationalTrustAssessor(
            {
                "feature": [1, 2, 3],
            }
        )


def test_assessor_rejects_empty_dataframe():
    """Current data must not be empty."""
    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        OperationalTrustAssessor(pd.DataFrame())


def test_assessor_accepts_reference_dataframe():
    """A valid reference dataset should be preserved as context."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30],
        }
    )

    reference_data = pd.DataFrame(
        {
            "feature": [5, 15, 25, 35],
        }
    )

    assessor = OperationalTrustAssessor(
        data,
        reference_data=reference_data,
    )

    assert assessor.reference_data_available is True
    assert assessor.reference_row_count == 4
    assert assessor.reference_column_count == 1


def test_assessor_rejects_non_dataframe_reference_data():
    """Reference data must be a DataFrame when supplied."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        TypeError,
        match="reference_data must be a pandas DataFrame",
    ):
        OperationalTrustAssessor(
            data,
            reference_data={"feature": [1, 2, 3]},
        )


def test_assessor_rejects_empty_reference_dataframe():
    """Reference data must not be empty when supplied."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="reference_data must not be empty",
    ):
        OperationalTrustAssessor(
            data,
            reference_data=pd.DataFrame(),
        )


def test_assessment_context_without_reference_data():
    """Context should report when no comparison dataset exists."""
    data = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": [3, 4],
        }
    )

    assessor = OperationalTrustAssessor(data)

    context = assessor.assessment_context()

    assert context["current_data"] == {
        "rows": 2,
        "columns": 2,
    }

    assert context["reference_data_available"] is False
    assert context["reference_data"] is None


def test_assessment_context_with_reference_data():
    """Context should describe both current and reference data."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30],
        }
    )

    reference_data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "other": ["A", "B", "C", "D", "E"],
        }
    )

    assessor = OperationalTrustAssessor(
        data,
        reference_data=reference_data,
    )

    context = assessor.assessment_context()

    assert context["current_data"] == {
        "rows": 3,
        "columns": 1,
    }

    assert context["reference_data_available"] is True

    assert context["reference_data"] == {
        "rows": 5,
        "columns": 2,
    }


def test_drift_availability_requires_reference_data():
    """Drift analysis should require a comparison dataset."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    without_reference = OperationalTrustAssessor(data)
    unavailable = without_reference.analysis_availability()

    assert unavailable["data_drift"]["available"] is False
    assert unavailable["data_drift"]["requires"] == [
        "current_data",
        "reference_data",
    ]

    reference_data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with_reference = OperationalTrustAssessor(
        data,
        reference_data=reference_data,
    )

    available = with_reference.analysis_availability()

    assert available["data_drift"]["available"] is True


def test_non_drift_foundation_analyses_are_available():
    """Foundation should expose the remaining planned analyses."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    availability = OperationalTrustAssessor(
        data
    ).analysis_availability()

    assert availability["data_stability"]["available"] is True
    assert availability["reproducibility"]["available"] is True
    assert availability["monitoring_readiness"]["available"] is True


def test_assess_returns_foundational_structure():
    """Foundational assessment should return expected sections."""
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": ["A", "B", "C"],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    assert set(report.keys()) == {
        "dataset",
        "reference_dataset",
        "assessment_context",
        "analysis_availability",
        "operational_trust_summary",
    }

    assert report["dataset"] == {
        "rows": 3,
        "columns": 2,
    }

    assert report["reference_dataset"] is None

    assert report["operational_trust_summary"] == {
        "status": "foundation",
        "review_required": False,
        "review_reasons": [],
    }


def test_assess_reports_reference_dataset():
    """Integrated output should describe supplied reference data."""
    data = pd.DataFrame(
        {
            "feature": [10, 20],
        }
    )

    reference_data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reference_data=reference_data,
    ).assess()

    assert report["reference_dataset"] == {
        "rows": 3,
        "columns": 1,
    }

    assert (
        report["assessment_context"]["reference_data_available"]
        is True
    )
