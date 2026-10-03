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


def test_assessor_rejects_invalid_representation_threshold():
    """Representation threshold should be within the supported range."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "target": [1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="representation_threshold",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
            sensitive_attributes=["group"],
            representation_threshold=0,
        )

    with pytest.raises(
        ValueError,
        match="representation_threshold",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
            sensitive_attributes=["group"],
            representation_threshold=1.1,
        )


def test_assessor_rejects_invalid_outcome_ratio_threshold():
    """Outcome-ratio threshold should be within the supported range."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "target": [1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="outcome_ratio_threshold",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
            sensitive_attributes=["group"],
            outcome_ratio_threshold=0,
        )

    with pytest.raises(
        ValueError,
        match="outcome_ratio_threshold",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
            sensitive_attributes=["group"],
            outcome_ratio_threshold=1.1,
        )


def test_assessor_rejects_invalid_performance_gap_threshold():
    """Performance-gap threshold should be between zero and one."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "target": [1, 0],
            "prediction": [1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="performance_gap_threshold",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
            prediction_column="prediction",
            sensitive_attributes=["group"],
            performance_gap_threshold=-0.01,
        )

    with pytest.raises(
        ValueError,
        match="performance_gap_threshold",
    ):
        ResponsibleAIAssessor(
            data,
            target_column="target",
            prediction_column="prediction",
            sensitive_attributes=["group"],
            performance_gap_threshold=1.01,
        )


def test_zero_performance_gap_threshold_is_allowed():
    """A zero performance-gap threshold should be supported."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "target": [1, 0],
            "prediction": [1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        performance_gap_threshold=0,
    )

    assert assessor.performance_gap_threshold == 0.0


def test_assessor_rejects_invalid_minimum_group_size():
    """Minimum group size should be a positive integer."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "target": [1, 0],
        }
    )

    for value in [0, -1, 2.5, True]:
        with pytest.raises(
            ValueError,
            match="minimum_group_size",
        ):
            ResponsibleAIAssessor(
                data,
                target_column="target",
                sensitive_attributes=["group"],
                minimum_group_size=value,
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

    assert availability["explainability"]["available"] is False


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

    assert availability["explainability"]["available"] is False


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
            "region": ["north", "south"] * 10,
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

    assert (
        summary["prediction_performance_fairness_applicable"]
        is False
    )


def test_no_fairness_disparity_does_not_add_fairness_review_reason():
    """No flagged disparity should not add a group-fairness review reason."""
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

    assert "group_fairness" not in summary["review_reasons"]
    assert summary["fairness_attributes_requiring_review"] == []

    assert report["explainability"]["applicable"] is False

    assert (
        report["explainability"]["explanation_readiness"]["status"]
        == "insufficient_features"
    )

    assert "explainability" in summary["review_reasons"]
    assert summary["status"] == "review"
    assert summary["review_required"] is True


def test_bias_indicators_not_applicable_without_target():
    """Bias analysis should be unavailable without observed outcomes."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "prediction": [1, 0, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    result = assessor.bias_indicator_analysis()

    assert result["applicable"] is False
    assert result["attributes"] == {}
    assert result["review_required"] is False
    assert result["attributes_requiring_review"] == []


def test_bias_indicators_not_applicable_without_sensitive_attributes():
    """Bias analysis should require at least one sensitive attribute."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "target": [1, 0, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
    )

    result = assessor.bias_indicator_analysis()

    assert result["applicable"] is False
    assert result["attributes"] == {}
    assert result["review_required"] is False
    assert result["attributes_requiring_review"] == []


def test_bias_indicators_are_integrated_for_single_attribute():
    """Integrated assessor should analyse observed outcomes by group."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "target": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        sensitive_attributes=["group"],
        minimum_group_size=5,
    )

    result = assessor.bias_indicator_analysis()

    assert result["applicable"] is True
    assert result["attributes_analyzed"] == 1
    assert "group" in result["attributes"]

    group_result = result["attributes"]["group"]

    assert group_result["reference_group"] == "A"

    assert (
        group_result["outcome_comparisons"]["B"][
            "outcome_rate_ratio"
        ]
        == 0.5
    )

    assert (
        group_result["outcome_comparisons"]["B"][
            "outcome_disparity"
        ]
        is True
    )

    assert result["attributes_requiring_review"] == ["group"]
    assert result["review_required"] is True


def test_bias_analysis_supports_multiple_sensitive_attributes():
    """Each sensitive attribute should receive bias analysis."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "region": ["north", "south"] * 10,
            "target": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        sensitive_attributes=["group", "region"],
        minimum_group_size=5,
    )

    result = assessor.bias_indicator_analysis()

    assert result["applicable"] is True
    assert result["attributes_analyzed"] == 2
    assert "group" in result["attributes"]
    assert "region" in result["attributes"]


def test_bias_thresholds_flow_into_integrated_analysis():
    """Configured bias thresholds should reach specialised analyzers."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 18 + ["B"] * 2,
            "target": (
                [1] * 14
                + [0] * 4
                + [0, 0]
            ),
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        sensitive_attributes=["group"],
        representation_threshold=0.20,
        outcome_ratio_threshold=0.90,
        minimum_group_size=5,
    )

    result = assessor.bias_indicator_analysis()

    assert result["representation_threshold"] == 0.20
    assert result["outcome_ratio_threshold"] == 0.90
    assert result["minimum_group_size"] == 5

    group_result = result["attributes"]["group"]

    assert group_result["group_metrics"]["B"]["underrepresented"] is True
    assert group_result["group_metrics"]["B"]["small_group"] is True


def test_bias_review_flows_into_responsible_ai_summary():
    """Bias indicators should contribute to the integrated summary."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "target": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
        }
    )

    report = ResponsibleAIAssessor(
        data,
        target_column="target",
        sensitive_attributes=["group"],
        minimum_group_size=5,
    ).assess()

    bias = report["bias_indicators"]
    summary = report["responsible_ai_summary"]

    assert bias["applicable"] is True
    assert bias["review_required"] is True

    assert summary["status"] == "review"
    assert summary["review_required"] is True
    assert "bias_indicators" in summary["review_reasons"]

    assert (
        summary["bias_attributes_requiring_review"]
        == ["group"]
    )

    assert (
        summary["prediction_performance_fairness_applicable"]
        is False
    )


def test_fairness_and_bias_can_both_require_review():
    """Prediction and observed-outcome disparities remain distinct."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "target": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
            "prediction": (
                [1] * 9
                + [0]
                + [1] * 3
                + [0] * 7
            ),
        }
    )

    report = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        minimum_group_size=5,
    ).assess()

    assert report["group_fairness"]["review_required"] is True
    assert report["bias_indicators"]["review_required"] is True

    summary = report["responsible_ai_summary"]

    assert "group_fairness" in summary["review_reasons"]
    assert "bias_indicators" in summary["review_reasons"]
    assert summary["review_required"] is True


def test_performance_fairness_not_applicable_without_target():
    """Performance fairness should be unavailable without a target."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "prediction": [1, 0, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    result = assessor.prediction_performance_fairness_analysis()

    assert result["applicable"] is False
    assert result["attributes"] == {}
    assert result["review_required"] is False
    assert result["attributes_requiring_review"] == []


def test_performance_fairness_not_applicable_without_predictions():
    """Performance fairness should be unavailable without predictions."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "target": [1, 0, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        sensitive_attributes=["group"],
    )

    result = assessor.prediction_performance_fairness_analysis()

    assert result["applicable"] is False
    assert result["attributes"] == {}
    assert result["review_required"] is False
    assert result["attributes_requiring_review"] == []


def test_performance_fairness_not_applicable_without_sensitive_attributes():
    """Performance fairness should require sensitive attributes."""
    data = pd.DataFrame(
        {
            "target": [1, 0, 1, 0],
            "prediction": [1, 0, 0, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
    )

    result = assessor.prediction_performance_fairness_analysis()

    assert result["applicable"] is False
    assert result["attributes"] == {}
    assert result["review_required"] is False
    assert result["attributes_requiring_review"] == []


def test_performance_fairness_is_integrated_for_single_attribute():
    """Integrated assessor should compare model performance by group."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "target": (
                [1, 1, 1, 0, 0, 0]
                + [1, 1, 1, 0, 0, 0]
            ),
            "prediction": (
                [1, 1, 1, 0, 0, 0]
                + [1, 0, 0, 1, 0, 0]
            ),
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        minimum_group_size=5,
    )

    result = assessor.prediction_performance_fairness_analysis()

    assert result["applicable"] is True
    assert result["attributes_analyzed"] == 1
    assert "group" in result["attributes"]

    group_result = result["attributes"]["group"]

    assert group_result["group_metrics"]["A"]["accuracy"] == 1.0

    assert (
        group_result["group_metrics"]["B"]["accuracy"]
        == 0.5
    )

    assert group_result["review_required"] is True
    assert result["attributes_requiring_review"] == ["group"]
    assert result["review_required"] is True


def test_performance_fairness_supports_multiple_sensitive_attributes():
    """Each sensitive attribute should receive performance analysis."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "region": ["north", "south"] * 6,
            "target": [
                1, 1, 1, 0, 0, 0,
                1, 1, 1, 0, 0, 0,
            ],
            "prediction": [
                1, 1, 1, 0, 0, 0,
                1, 0, 0, 1, 0, 0,
            ],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group", "region"],
        minimum_group_size=1,
    )

    result = assessor.prediction_performance_fairness_analysis()

    assert result["applicable"] is True
    assert result["attributes_analyzed"] == 2
    assert "group" in result["attributes"]
    assert "region" in result["attributes"]


def test_performance_threshold_flows_into_integrated_analysis():
    """Configured performance threshold should reach the analyzer."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "target": [
                1, 1, 1, 0, 0, 0,
                1, 1, 1, 0, 0, 0,
            ],
            "prediction": [
                1, 1, 1, 0, 0, 0,
                1, 0, 0, 1, 0, 0,
            ],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        performance_gap_threshold=0.20,
        minimum_group_size=3,
    )

    result = assessor.prediction_performance_fairness_analysis()

    assert result["performance_gap_threshold"] == 0.20
    assert result["minimum_group_size"] == 3

    group_result = result["attributes"]["group"]

    assert group_result["performance_gap_threshold"] == 0.20
    assert group_result["minimum_group_size"] == 3


def test_performance_review_flows_into_responsible_ai_summary():
    """Performance disparities should contribute to integrated review."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "target": [
                1, 1, 1, 0, 0, 0,
                1, 1, 1, 0, 0, 0,
            ],
            "prediction": [
                1, 1, 1, 0, 0, 0,
                1, 0, 0, 1, 0, 0,
            ],
        }
    )

    report = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        performance_gap_threshold=0.10,
        minimum_group_size=5,
    ).assess()

    performance = report["prediction_performance_fairness"]
    summary = report["responsible_ai_summary"]

    assert performance["applicable"] is True
    assert performance["review_required"] is True

    assert summary["status"] == "review"
    assert summary["review_required"] is True

    assert (
        "prediction_performance_fairness"
        in summary["review_reasons"]
    )

    assert (
        summary["performance_attributes_requiring_review"]
        == ["group"]
    )

    assert (
        summary["prediction_performance_fairness_applicable"]
        is True
    )


def test_equal_performance_does_not_add_performance_review_reason():
    """Equal group performance should not add a performance reason."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "target": [
                1, 1, 1, 0, 0, 0,
                1, 1, 1, 0, 0, 0,
            ],
            "prediction": [
                1, 1, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0,
            ],
        }
    )

    report = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        minimum_group_size=5,
    ).assess()

    performance = report["prediction_performance_fairness"]
    summary = report["responsible_ai_summary"]

    assert performance["review_required"] is False

    assert (
        "prediction_performance_fairness"
        not in summary["review_reasons"]
    )

    assert (
        summary["performance_attributes_requiring_review"]
        == []
    )


def test_all_three_responsible_ai_layers_can_require_review():
    """Fairness, bias and performance indicators can coexist."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "target": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
            "prediction": (
                [1] * 8
                + [0] * 2
                + [1] * 2
                + [0] * 8
            ),
        }
    )

    report = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        minimum_group_size=5,
    ).assess()

    assert report["group_fairness"]["review_required"] is True
    assert report["bias_indicators"]["review_required"] is True

    assert (
        report["prediction_performance_fairness"][
            "review_required"
        ]
        is True
    )

    summary = report["responsible_ai_summary"]

    assert summary["status"] == "review"
    assert summary["review_required"] is True

    assert "group_fairness" in summary["review_reasons"]
    assert "bias_indicators" in summary["review_reasons"]

    assert (
        "prediction_performance_fairness"
        in summary["review_reasons"]
    )

    assert summary["fairness_attributes_requiring_review"] == [
        "group"
    ]

    assert summary["bias_attributes_requiring_review"] == [
        "group"
    ]

    assert summary["performance_attributes_requiring_review"] == [
        "group"
    ]


def test_custom_positive_label_flows_into_performance_analysis():
    """Configured positive label should reach performance analysis."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "target": [
                "approved",
                "approved",
                "denied",
                "denied",
                "approved",
                "approved",
                "denied",
                "denied",
            ],
            "prediction": [
                "approved",
                "approved",
                "denied",
                "denied",
                "approved",
                "denied",
                "approved",
                "denied",
            ],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        positive_label="approved",
        minimum_group_size=1,
    )

    result = assessor.prediction_performance_fairness_analysis()

    assert result["positive_label"] == "approved"

    group_result = result["attributes"]["group"]

    assert group_result["positive_label"] == "approved"

    assert (
        group_result["group_metrics"]["A"]["accuracy"]
        == 1.0
    )

    assert (
        group_result["group_metrics"]["B"]["accuracy"]
        == 0.5
    )


def test_performance_limited_evidence_flows_into_review():
    """Small groups should remain visible in integrated analysis."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 2,
            "target": [
                1, 1, 1, 1, 0, 0, 0, 0,
                1, 0,
            ],
            "prediction": [
                1, 1, 1, 1, 0, 0, 0, 0,
                1, 0,
            ],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        minimum_group_size=5,
    )

    result = assessor.prediction_performance_fairness_analysis()

    group_result = result["attributes"]["group"]

    assert (
        group_result["group_metrics"]["B"]["limited_evidence"]
        is True
    )

    assert (
        "B"
        in group_result["review_summary"][
            "limited_evidence_groups"
        ]
    )

    assert result["review_required"] is True
    assert result["attributes_requiring_review"] == ["group"]


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
        "bias_indicators",
        "prediction_performance_fairness",
        "explainability",
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
    assert report["bias_indicators"]["applicable"] is True

    assert (
        report["prediction_performance_fairness"]["applicable"]
        is True
    )

    assert (
        report["responsible_ai_summary"][
            "group_fairness_applicable"
        ]
        is True
    )

    assert (
        report["responsible_ai_summary"][
            "bias_indicators_applicable"
        ]
        is True
    )

    assert (
        report["responsible_ai_summary"][
            "prediction_performance_fairness_applicable"
        ]
        is True
    )


    assert report["explainability"]["applicable"] is True

    assert report["explainability"]["candidate_features"] == [
        "feature"
    ]

    assert (
        report["responsible_ai_summary"]["explainability_applicable"]
        is True
    )

    assert (
        report["responsible_ai_summary"]["explanation_readiness_status"]
        == "ready_for_explanation_analysis"
    )


def test_multiple_sensitive_attributes_are_supported():
    """Assessor should support multiple sensitive attributes."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "region": ["north", "south", "north", "south"],
            "target": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group", "region"],
        minimum_group_size=1,
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

    bias = assessor.bias_indicator_analysis()

    assert bias["applicable"] is True
    assert bias["attributes_analyzed"] == 2
    assert "group" in bias["attributes"]
    assert "region" in bias["attributes"]

    performance = (
        assessor.prediction_performance_fairness_analysis()
    )

    assert performance["applicable"] is True
    assert performance["attributes_analyzed"] == 2
    assert "group" in performance["attributes"]
    assert "region" in performance["attributes"]


def test_assessor_rejects_invalid_high_cardinality_threshold():
    """High-cardinality threshold should be a positive integer."""
    data = pd.DataFrame(
        {
            "feature": ["A", "B"],
        }
    )

    for value in [0, -1, 2.5, True]:
        with pytest.raises(
            ValueError,
            match="high_cardinality_threshold",
        ):
            ResponsibleAIAssessor(
                data,
                high_cardinality_threshold=value,
            )


def test_assessor_rejects_invalid_missingness_threshold():
    """Explainability missingness threshold should be between zero and one."""
    data = pd.DataFrame(
        {
            "feature": [1, 2],
        }
    )

    for value in [-0.01, 1.01]:
        with pytest.raises(
            ValueError,
            match="missingness_threshold",
        ):
            ResponsibleAIAssessor(
                data,
                missingness_threshold=value,
            )


def test_explainability_threshold_boundaries_are_allowed():
    """Valid explainability threshold boundaries should be preserved."""
    data = pd.DataFrame(
        {
            "feature": [1, 2],
        }
    )

    zero = ResponsibleAIAssessor(
        data,
        missingness_threshold=0,
        high_cardinality_threshold=1,
    )

    one = ResponsibleAIAssessor(
        data,
        missingness_threshold=1,
    )

    assert zero.missingness_threshold == 0.0
    assert zero.high_cardinality_threshold == 1
    assert one.missingness_threshold == 1.0


def test_explainability_is_available_without_predictions():
    """Structural explainability readiness should not require predictions."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "income": [30000, 40000, 50000, 60000],
        }
    )

    assessor = ResponsibleAIAssessor(data)

    availability = assessor.analysis_availability()
    result = assessor.explainability_analysis()

    assert availability["explainability"]["available"] is True
    assert availability["explainability"]["requires"] == [
        "candidate_explanatory_features"
    ]

    assert result["applicable"] is True
    assert result["candidate_features"] == ["age", "income"]

    assert (
        result["explanation_readiness"]["status"]
        == "ready_for_explanation_analysis"
    )


def test_explainability_excludes_context_columns():
    """Target, prediction and sensitive attributes should be excluded."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "income": [30000, 40000, 50000, 60000],
            "group": ["A", "A", "B", "B"],
            "target": [1, 0, 1, 0],
            "prediction": [1, 0, 0, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    assert assessor.explainability_candidate_features() == [
        "age",
        "income",
    ]

    result = assessor.explainability_analysis()

    assert result["applicable"] is True

    assert result["excluded_columns"] == [
        "target",
        "prediction",
        "group",
    ]

    assert result["candidate_features"] == [
        "age",
        "income",
    ]


def test_explainability_thresholds_flow_into_integrated_analysis():
    """Configured explainability thresholds should reach the analyzer."""
    data = pd.DataFrame(
        {
            "category": ["A", "B", "C", "D"],
            "feature": [1, None, None, 4],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        high_cardinality_threshold=3,
        missingness_threshold=0.50,
    )

    result = assessor.explainability_analysis()

    assert result["high_cardinality_threshold"] == 3
    assert result["missingness_threshold"] == 0.50

    assert (
        result["feature_analysis"]["category"]["high_cardinality"]
        is True
    )

    assert (
        result["feature_analysis"]["feature"]["high_missingness"]
        is True
    )


def test_explainability_review_flows_into_responsible_ai_summary():
    """Explainability indicators should contribute to integrated review."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "constant": [1, 1, 1, 1],
        }
    )

    report = ResponsibleAIAssessor(data).assess()

    explainability = report["explainability"]
    summary = report["responsible_ai_summary"]

    assert explainability["applicable"] is True
    assert explainability["review_required"] is True

    assert (
        explainability["explanation_readiness"]["status"]
        == "review"
    )

    assert summary["status"] == "review"
    assert summary["review_required"] is True
    assert "explainability" in summary["review_reasons"]

    assert (
        summary["explainability_features_requiring_review"]
        == ["constant"]
    )

    assert summary["explanation_readiness_status"] == "review"


def test_clean_explainability_does_not_add_review_reason():
    """Clean explanatory features should not add an explainability reason."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "income": [30000, 40000, 50000, 60000],
        }
    )

    report = ResponsibleAIAssessor(data).assess()

    explainability = report["explainability"]
    summary = report["responsible_ai_summary"]

    assert explainability["review_required"] is False

    assert (
        explainability["explanation_readiness"]["status"]
        == "ready_for_explanation_analysis"
    )

    assert "explainability" not in summary["review_reasons"]

    assert (
        summary["explainability_features_requiring_review"]
        == []
    )

    assert summary["status"] == "no_review_indicators"
    assert summary["review_required"] is False


def test_no_candidate_features_are_reported_as_insufficient():
    """Context-only data should expose insufficient explanation readiness."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "target": [1, 0, 1, 0],
            "prediction": [1, 0, 0, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        minimum_group_size=1,
    )

    availability = assessor.analysis_availability()
    report = assessor.assess()

    explainability = report["explainability"]
    summary = report["responsible_ai_summary"]

    assert availability["explainability"]["available"] is False

    assert explainability["applicable"] is False
    assert explainability["candidate_features"] == []
    assert explainability["candidate_feature_count"] == 0

    assert (
        explainability["explanation_readiness"]["status"]
        == "insufficient_features"
    )

    assert explainability["review_required"] is True
    assert "explainability" in summary["review_reasons"]

    assert summary["explainability_applicable"] is False

    assert (
        summary["explanation_readiness_status"]
        == "insufficient_features"
    )


def test_explainability_supports_multiple_sensitive_attributes():
    """All configured sensitive attributes should be excluded as context."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "region": ["north", "south", "north", "south"],
            "age": [25, 35, 45, 55],
            "income": [30000, 40000, 50000, 60000],
            "target": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 0],
        }
    )

    assessor = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group", "region"],
        minimum_group_size=1,
    )

    result = assessor.explainability_analysis()

    assert result["applicable"] is True

    assert result["excluded_columns"] == [
        "target",
        "prediction",
        "group",
        "region",
    ]

    assert result["candidate_features"] == [
        "age",
        "income",
    ]

    assert "group" not in result["feature_analysis"]
    assert "region" not in result["feature_analysis"]


def test_all_four_responsible_ai_layers_can_contribute_to_review():
    """All implemented Responsible AI layers can coexist in review."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "constant_feature": [1] * 20,
            "target": (
                [1] * 8
                + [0] * 2
                + [1] * 4
                + [0] * 6
            ),
            "prediction": (
                [1] * 8
                + [0] * 2
                + [1] * 2
                + [0] * 8
            ),
        }
    )

    report = ResponsibleAIAssessor(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        minimum_group_size=5,
    ).assess()

    assert report["group_fairness"]["review_required"] is True
    assert report["bias_indicators"]["review_required"] is True

    assert (
        report["prediction_performance_fairness"]["review_required"]
        is True
    )

    assert report["explainability"]["review_required"] is True

    summary = report["responsible_ai_summary"]

    assert "group_fairness" in summary["review_reasons"]
    assert "bias_indicators" in summary["review_reasons"]

    assert (
        "prediction_performance_fairness"
        in summary["review_reasons"]
    )

    assert "explainability" in summary["review_reasons"]

    assert (
        summary["explainability_features_requiring_review"]
        == ["constant_feature"]
    )

