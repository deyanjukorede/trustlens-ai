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


def test_assessor_rejects_invalid_disparate_impact_threshold():
    """Fairness threshold should be within the supported range."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "prediction": [1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="disparate_impact_threshold",
    ):
        ResponsibleAIAssessor(
            data,
            prediction_column="prediction",
            sensitive_attributes=["group"],
            disparate_impact_threshold=0,
        )

    with pytest.raises(
        ValueError,
        match="disparate_impact_threshold",
    ):
        ResponsibleAIAssessor(
            data,
            prediction_column="prediction",
            sensitive_attributes=["group"],
            disparate_impact_threshold=1.1,
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


def test_group_fairness_not_applicable_without_predictions():
    """Fairness analysis should be unavailable without predictions."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        sensitive_attributes=["group"],
    )

    result = assessor.group_fairness_analysis()

    assert result["applicable"] is False
    assert result["attributes"] == {}
    assert result["review_required"] is False
    assert result["attributes_requiring_review"] == []


def test_group_fairness_not_applicable_without_sensitive_attributes():
    """Fairness analysis should require a sensitive attribute."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "prediction": [1, 0, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
    )

    result = assessor.group_fairness_analysis()

    assert result["applicable"] is False
    assert result["attributes"] == {}
    assert result["review_required"] is False
    assert result["attributes_requiring_review"] == []


def test_group_fairness_is_integrated_for_single_attribute():
    """Integrated assessor should analyse one sensitive attribute."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "prediction": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    result = assessor.group_fairness_analysis()

    assert result["applicable"] is True
    assert result["attributes_analyzed"] == 1
    assert "group" in result["attributes"]

    group_result = result["attributes"]["group"]

    assert group_result["reference_group"] == "A"
    assert (
        group_result["comparisons"]["B"]["disparate_impact_ratio"]
        == 0.5
    )
    assert group_result["review_required"] is True

    assert result["attributes_requiring_review"] == ["group"]
    assert result["review_required"] is True


def test_multiple_sensitive_attributes_receive_fairness_analysis():
    """Each configured sensitive attribute should be analysed."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "region": (
                ["north", "south"] * 10
            ),
            "prediction": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group", "region"],
    )

    result = assessor.group_fairness_analysis()

    assert result["applicable"] is True
    assert result["attributes_analyzed"] == 2
    assert "group" in result["attributes"]
    assert "region" in result["attributes"]


def test_custom_positive_label_flows_into_fairness_analysis():
    """Configured positive label should be used by fairness analysis."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "decision": [
                "approved",
                "denied",
                "approved",
                "approved",
            ],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="decision",
        sensitive_attributes=["group"],
        positive_label="approved",
    )

    result = assessor.group_fairness_analysis()

    assert result["positive_label"] == "approved"

    group_result = result["attributes"]["group"]

    assert group_result["positive_label"] == "approved"
    assert group_result["group_metrics"]["A"]["positive_count"] == 1
    assert group_result["group_metrics"]["B"]["positive_count"] == 2


def test_custom_disparate_impact_threshold_is_used():
    """Configured disparity threshold should flow into analysis."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "prediction": (
                [1] * 8
                + [0] * 2
                + [1] * 7
                + [0] * 3
            ),
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group"],
        disparate_impact_threshold=0.90,
    )

    result = assessor.group_fairness_analysis()

    assert result["disparate_impact_threshold"] == 0.90

    group_result = result["attributes"]["group"]

    assert (
        group_result["comparisons"]["B"]["disparate_impact_ratio"]
        == 0.875
    )
    assert group_result["comparisons"]["B"]["requires_review"] is True


def test_fairness_review_flows_into_responsible_ai_summary():
    """Fairness disparities should appear in the integrated summary."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "prediction": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
        }
    )

    report = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group"],
    ).assess()

    summary = report["responsible_ai_summary"]

    assert summary["status"] == "review"
    assert summary["review_required"] is True
    assert "group_fairness" in summary["review_reasons"]

    assert (
        summary["fairness_attributes_requiring_review"]
        == ["group"]
    )


def test_no_fairness_disparity_produces_no_review_indicator():
    """No flagged disparity should produce no review indicator."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "prediction": (
                [1] * 8
                + [0] * 2
                + [1] * 7
                + [0] * 3
            ),
        }
    )

    report = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group"],
    ).assess()

    fairness = report["group_fairness"]
    summary = report["responsible_ai_summary"]

    assert fairness["review_required"] is False
    assert fairness["attributes_requiring_review"] == []

    assert summary["status"] == "no_review_indicators"
    assert summary["review_required"] is False
    assert summary["review_reasons"] == []


def test_assess_returns_expected_structure():
    """Integrated assessment should return expected sections."""
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
        "group_fairness",
        "responsible_ai_summary",
    }

    assert set(report.keys()) == expected_sections

    assert report["dataset"] == {
        "rows": 4,
        "columns": 4,
    }

    assert report["target_column"] == "target"
    assert report["prediction_column"] == "prediction"
    assert report["sensitive_attributes"] == ["group"]

    assert report["group_fairness"]["applicable"] is True

    assert (
        report["responsible_ai_summary"][
            "group_fairness_applicable"
        ]
        is True
    )


def test_multiple_sensitive_attributes_are_supported():
    """Assessor should support multiple sensitive attributes."""
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

    fairness = assessor.group_fairness_analysis()

    assert fairness["applicable"] is True
    assert fairness["attributes_analyzed"] == 2
    assert "group" in fairness["attributes"]
    assert "region" in fairness["attributes"]
