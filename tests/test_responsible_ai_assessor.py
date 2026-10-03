import pandas as pd
import pytest

from trustlens.responsible_ai.assessor import ResponsibleAIAssessor


def test_assessor_initialises_with_valid_dataframe():
    """Assessor should initialise with a valid non-empty DataFrame."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "target": [0, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
    )

    assert assessor.row_count == 3
    assert assessor.column_count == 2
    assert assessor.target_column == "target"


def test_assessor_rejects_non_dataframe_input():
    """Assessor should reject inputs that are not pandas DataFrames."""
    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
        ResponsibleAIAssessor(
            {
                "age": [25, 35],
            }
        )


def test_assessor_rejects_empty_dataframe():
    """Assessor should reject an empty DataFrame."""
    data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        ResponsibleAIAssessor(data)


def test_assessor_rejects_missing_target_column():
    """Assessor should validate a supplied target column."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45],
        }
    )

    with pytest.raises(
        ValueError,
        match="target_column",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
        )


def test_assessor_rejects_missing_prediction_column():
    """Assessor should validate a supplied prediction column."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "target": [0, 1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="prediction_column",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
            prediction_column="prediction",
        )


def test_assessor_rejects_non_list_sensitive_attributes():
    """Sensitive attributes should be supplied as a list."""
    data = pd.DataFrame(
        {
            "gender": ["A", "B", "A"],
            "target": [0, 1, 0],
        }
    )

    with pytest.raises(
        TypeError,
        match="sensitive_attributes must be a list",
    ):
        ResponsibleAIAssessor(
            data,
            sensitive_attributes="gender",
        )


def test_assessor_rejects_missing_sensitive_attribute():
    """Configured sensitive attributes must exist in the dataset."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "target": [0, 1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="sensitive attributes",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
            sensitive_attributes=["gender"],
        )


def test_assessment_context_with_complete_information():
    """Context should identify available Responsible AI inputs."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 1],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    context = assessor.assessment_context()

    assert context["target_available"] is True
    assert context["predictions_available"] is True
    assert context["sensitive_attributes_available"] is True
    assert context["sensitive_attribute_count"] == 1


def test_assessment_context_without_optional_information():
    """Optional modelling context should not be required."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30],
        }
    )

    assessor = ResponsibleAIAssessor(data)

    context = assessor.assessment_context()

    assert context["target_available"] is False
    assert context["predictions_available"] is False
    assert context["sensitive_attributes_available"] is False
    assert context["sensitive_attribute_count"] == 0


def test_sensitive_attribute_summary():
    """Sensitive attribute summary should report group structure."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", None],
            "target": [0, 1, 0, 1],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        sensitive_attributes=["group"],
    )

    summary = assessor.sensitive_attribute_summary()

    assert summary["attribute_count"] == 1
    assert "group" in summary["attributes"]

    group_summary = summary["attributes"]["group"]

    assert group_summary["unique_values"] == 2
    assert group_summary["missing_values"] == 1
    assert group_summary["group_counts"]["A"] == 2
    assert group_summary["group_counts"]["B"] == 1


def test_group_fairness_availability():
    """Group fairness should require sensitive groups and predictions."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "prediction": [0, 1, 1, 1],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    availability = assessor.analysis_availability()

    assert availability["group_fairness"]["available"] is True
    assert availability["outcome_bias"]["available"] is False
    assert (
        availability["prediction_performance_fairness"]["available"]
        is False
    )
    assert availability["explainability"]["available"] is True


def test_prediction_performance_fairness_availability():
    """Performance fairness should require groups, target and predictions."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 1],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    availability = assessor.analysis_availability()

    assert availability["group_fairness"]["available"] is True
    assert availability["outcome_bias"]["available"] is True
    assert (
        availability["prediction_performance_fairness"]["available"]
        is True
    )
    assert availability["explainability"]["available"] is True


def test_assess_returns_expected_structure():
    """Full foundational assessment should return expected sections."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "feature": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 1],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    report = assessor.assess()

    expected_sections = {
        "dataset",
        "target_column",
        "prediction_column",
        "sensitive_attributes",
        "assessment_context",
        "sensitive_attribute_summary",
        "analysis_availability",
    }

    assert set(report.keys()) == expected_sections

    assert report["dataset"] == {
        "rows": 4,
        "columns": 4,
    }

    assert report["target_column"] == "target"
    assert report["prediction_column"] == "prediction"
    assert report["sensitive_attributes"] == ["group"]


def test_multiple_sensitive_attributes_are_supported():
    """Assessor should support assessment across multiple attributes."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "region": ["north", "south", "north", "south"],
            "prediction": [0, 1, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group", "region"],
    )

    assert assessor.sensitive_attribute_count == 2

    summary = assessor.sensitive_attribute_summary()

    assert summary["attribute_count"] == 2
    assert "group" in summary["attributes"]
    assert "region" in summary["attributes"]
