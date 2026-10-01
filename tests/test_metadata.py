import pandas as pd
import pytest

from trustlens.governance.metadata import MetadataCompletenessAssessor


@pytest.fixture
def sample_data():
    """Create a representative dataset for metadata assessment."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "age": [28, 35, 42],
            "risk": ["low", "medium", "high"],
        }
    )


@pytest.fixture
def complete_metadata():
    """Create complete governance metadata for all sample columns."""
    return {
        "customer_id": {
            "description": "Unique identifier for each customer",
            "owner": "Customer Operations",
            "business_definition": "Internal customer identifier",
        },
        "age": {
            "description": "Customer age in years",
            "owner": "Customer Operations",
            "business_definition": "Age of customer at assessment",
        },
        "risk": {
            "description": "Customer risk classification",
            "owner": "Risk Management",
            "business_definition": "Assigned customer risk category",
        },
    }


@pytest.fixture
def partial_metadata():
    """Create partially completed governance metadata."""
    return {
        "customer_id": {
            "description": "Unique identifier for each customer",
            "owner": "Customer Operations",
            "business_definition": "Internal customer identifier",
        },
        "age": {
            "description": "Customer age in years",
            "owner": "",
            "business_definition": None,
        },
        "risk": {
            "description": "Customer risk classification",
        },
    }


def test_assessor_accepts_dataframe(sample_data):
    """Assessor should accept a valid pandas DataFrame."""
    assessor = MetadataCompletenessAssessor(sample_data)

    assert assessor.data.equals(sample_data)


def test_invalid_dataframe_input():
    """Assessor should reject input that is not a pandas DataFrame."""
    with pytest.raises(TypeError):
        MetadataCompletenessAssessor([1, 2, 3])


def test_empty_dataframe():
    """Assessor should reject an empty pandas DataFrame."""
    with pytest.raises(ValueError):
        MetadataCompletenessAssessor(pd.DataFrame())


def test_invalid_metadata_input(sample_data):
    """Assessor should reject metadata that is not a mapping."""
    with pytest.raises(TypeError):
        MetadataCompletenessAssessor(
            sample_data,
            metadata=["invalid"],
        )


def test_structural_metadata_contains_all_columns(sample_data):
    """Structural metadata should contain every dataset column."""
    assessor = MetadataCompletenessAssessor(sample_data)
    structural = assessor.structural_metadata()

    assert set(structural.keys()) == {
        "customer_id",
        "age",
        "risk",
    }


def test_structural_metadata_contains_name_and_type(sample_data):
    """Structural metadata should include column name and data type."""
    assessor = MetadataCompletenessAssessor(sample_data)
    structural = assessor.structural_metadata()

    for column in sample_data.columns:
        assert structural[column]["column_name"] == column
        assert "data_type" in structural[column]
        assert isinstance(structural[column]["data_type"], str)


def test_no_metadata_has_zero_score(sample_data):
    """Dataset without supplied metadata should score zero."""
    assessor = MetadataCompletenessAssessor(sample_data)

    assert assessor.completeness_score() == 0.0
    assert assessor.completeness_level() == "Missing"


def test_complete_metadata_scores_100(
    sample_data,
    complete_metadata,
):
    """Complete metadata should receive a score of 100."""
    assessor = MetadataCompletenessAssessor(
        sample_data,
        complete_metadata,
    )

    assert assessor.completeness_score() == 100.0
    assert assessor.completeness_level() == "Complete"


def test_partial_metadata_score(
    sample_data,
    partial_metadata,
):
    """Partially completed metadata should produce expected score."""
    assessor = MetadataCompletenessAssessor(
        sample_data,
        partial_metadata,
    )

    assert assessor.completeness_score() == 55.55
    assert assessor.completeness_level() == "Partial"


def test_column_metadata_status_complete(
    sample_data,
    complete_metadata,
):
    """Complete column metadata should report no missing fields."""
    assessor = MetadataCompletenessAssessor(
        sample_data,
        complete_metadata,
    )
    status = assessor.column_metadata_status()

    assert status["customer_id"]["completeness_score"] == 100.0
    assert status["customer_id"]["missing_fields"] == []
    assert status["customer_id"]["completed_fields"] == 3


def test_column_metadata_status_identifies_missing_fields(
    sample_data,
    partial_metadata,
):
    """Column assessment should identify missing metadata fields."""
    assessor = MetadataCompletenessAssessor(
        sample_data,
        partial_metadata,
    )
    status = assessor.column_metadata_status()

    assert status["age"]["missing_fields"] == [
        "owner",
        "business_definition",
    ]

    assert status["risk"]["missing_fields"] == [
        "owner",
        "business_definition",
    ]


def test_whitespace_metadata_is_missing(sample_data):
    """Whitespace-only metadata should be treated as missing."""
    metadata = {
        "customer_id": {
            "description": "   ",
            "owner": "\t",
            "business_definition": "\n",
        }
    }

    assessor = MetadataCompletenessAssessor(
        sample_data,
        metadata,
    )
    status = assessor.column_metadata_status()

    assert status["customer_id"]["completeness_score"] == 0.0


def test_none_metadata_value_is_missing(sample_data):
    """None metadata values should be treated as missing."""
    metadata = {
        "customer_id": {
            "description": None,
            "owner": "Customer Operations",
            "business_definition": "Customer identifier",
        }
    }

    assessor = MetadataCompletenessAssessor(
        sample_data,
        metadata,
    )
    status = assessor.column_metadata_status()

    assert status["customer_id"]["field_status"]["description"] is False


def test_invalid_column_metadata_mapping(sample_data):
    """Column metadata must itself be a mapping."""
    metadata = {
        "customer_id": "invalid metadata",
    }

    assessor = MetadataCompletenessAssessor(
        sample_data,
        metadata,
    )

    with pytest.raises(TypeError):
        assessor.column_metadata_status()


def test_recommendations_for_missing_metadata(sample_data):
    """Incomplete metadata should generate recommendations."""
    assessor = MetadataCompletenessAssessor(sample_data)

    recommendations = assessor.recommendations()

    assert isinstance(recommendations, list)
    assert len(recommendations) == 3
    assert all(
        "Complete metadata for" in recommendation
        for recommendation in recommendations
    )


def test_complete_metadata_recommendation(
    sample_data,
    complete_metadata,
):
    """Complete metadata should return a positive recommendation."""
    assessor = MetadataCompletenessAssessor(
        sample_data,
        complete_metadata,
    )

    assert assessor.recommendations() == [
        "Metadata is complete for all assessed dataset columns."
    ]


def test_assess_returns_expected_structure(
    sample_data,
    complete_metadata,
):
    """Complete assessment should contain expected report sections."""
    assessor = MetadataCompletenessAssessor(
        sample_data,
        complete_metadata,
    )
    report = assessor.assess()

    expected_keys = {
        "metadata_completeness_score",
        "metadata_completeness_level",
        "columns_assessed",
        "columns_with_complete_metadata",
        "columns_with_incomplete_metadata",
        "required_metadata_fields",
        "structural_metadata",
        "column_metadata",
        "recommendations",
    }

    assert set(report.keys()) == expected_keys


def test_assess_reports_column_counts(
    sample_data,
    complete_metadata,
):
    """Assessment should report metadata column counts correctly."""
    assessor = MetadataCompletenessAssessor(
        sample_data,
        complete_metadata,
    )
    report = assessor.assess()

    assert report["columns_assessed"] == 3
    assert report["columns_with_complete_metadata"] == 3
    assert report["columns_with_incomplete_metadata"] == 0


def test_assess_reports_required_fields(sample_data):
    """Assessment should expose required governance metadata fields."""
    assessor = MetadataCompletenessAssessor(sample_data)
    report = assessor.assess()

    assert report["required_metadata_fields"] == [
        "description",
        "owner",
        "business_definition",
    ]


def test_assessment_is_deterministic(
    sample_data,
    partial_metadata,
):
    """Repeated assessments should produce identical results."""
    assessor = MetadataCompletenessAssessor(
        sample_data,
        partial_metadata,
    )

    first_report = assessor.assess()
    second_report = assessor.assess()

    assert first_report == second_report
