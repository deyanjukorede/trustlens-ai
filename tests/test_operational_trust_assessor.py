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
    assert assessor.reproducibility_metadata == {}
    assert assessor.monitoring_metadata == {}


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


def test_assessor_rejects_non_dictionary_monitoring_metadata():
    """Monitoring metadata must be a dictionary when supplied."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        TypeError,
        match="monitoring_metadata must be a dictionary when supplied",
    ):
        OperationalTrustAssessor(
            data,
            monitoring_metadata=["monitoring"],
        )


def test_assessment_context_without_reference_data():
    """Context should report when optional context is absent."""
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
    assert context["reproducibility_metadata_supplied"] is False
    assert context["identifier_column"] is None
    assert context["monitoring_metadata_supplied"] is False


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


def test_assessment_context_reports_monitoring_metadata():
    """Context should report supplied monitoring metadata."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    assessor = OperationalTrustAssessor(
        data,
        monitoring_metadata={
            "monitoring_owner": "data-team",
        },
    )

    context = assessor.assessment_context()

    assert context["monitoring_metadata_supplied"] is True


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
    """Foundation should expose non-drift analyses."""
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


def test_monitoring_availability_reports_optional_context():
    """Monitoring availability should describe optional metadata."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    availability = OperationalTrustAssessor(
        data
    ).analysis_availability()

    assert availability["monitoring_readiness"] == {
        "available": True,
        "requires": ["current_data"],
        "optional_context": ["monitoring_metadata"],
    }


def test_assess_returns_integrated_structure_without_reference_data():
    """Assessment should expose integrated Operational Trust components."""
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": ["A", "B", "A"],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    assert set(report.keys()) == {
        "dataset",
        "reference_dataset",
        "assessment_context",
        "analysis_availability",
        "data_drift",
        "data_stability",
        "reproducibility",
        "monitoring_readiness",
        "operational_trust_overview",
        "operational_trust_summary",
    }

    assert report["dataset"] == {
        "rows": 3,
        "columns": 2,
    }

    assert report["reference_dataset"] is None
    assert report["data_drift"] is None

    assert report["data_stability"] is not None
    assert report["data_stability"]["review_required"] is False

    assert report["reproducibility"] is not None
    assert report["reproducibility"]["metadata_supplied"] is False
    assert report["reproducibility"]["review_required"] is True

    assert report["monitoring_readiness"] is not None
    assert report["monitoring_readiness"]["metadata_supplied"] is False
    assert report["monitoring_readiness"]["review_required"] is True

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": [
            "reproducibility",
            "monitoring_readiness",
        ],
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
