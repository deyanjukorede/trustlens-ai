import pandas as pd
import pytest

from trustlens.responsible_ai.fairness import FairnessAnalyzer


def test_fairness_analyzer_initialises_with_valid_data():
    """Analyzer should initialise with valid fairness inputs."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "prediction": [1, 0, 1, 1],
        }
    )

    analyzer = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    )

    assert analyzer.sensitive_attribute == "group"
    assert analyzer.prediction_column == "prediction"
    assert analyzer.positive_label == 1
    assert analyzer.disparate_impact_threshold == 0.80


def test_fairness_analyzer_rejects_non_dataframe():
    """Analyzer should reject non-DataFrame input."""
    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
        FairnessAnalyzer(
            {"group": ["A"], "prediction": [1]},
            sensitive_attribute="group",
            prediction_column="prediction",
        )


def test_fairness_analyzer_rejects_empty_dataframe():
    """Analyzer should reject an empty dataset."""
    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        FairnessAnalyzer(
            pd.DataFrame(),
            sensitive_attribute="group",
            prediction_column="prediction",
        )


def test_fairness_analyzer_validates_sensitive_attribute():
    """Sensitive attribute must exist in the dataset."""
    data = pd.DataFrame(
        {
            "prediction": [1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="sensitive_attribute",
    ):
        FairnessAnalyzer(
            data,
            sensitive_attribute="group",
            prediction_column="prediction",
        )


def test_fairness_analyzer_validates_prediction_column():
    """Prediction column must exist in the dataset."""
    data = pd.DataFrame(
        {
            "group": ["A", "B", "A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="prediction_column",
    ):
        FairnessAnalyzer(
            data,
            sensitive_attribute="group",
            prediction_column="prediction",
        )


def test_fairness_analyzer_validates_threshold():
    """Disparate-impact threshold must be within the valid range."""
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
        FairnessAnalyzer(
            data,
            sensitive_attribute="group",
            prediction_column="prediction",
            disparate_impact_threshold=0,
        )

    with pytest.raises(
        ValueError,
        match="disparate_impact_threshold",
    ):
        FairnessAnalyzer(
            data,
            sensitive_attribute="group",
            prediction_column="prediction",
            disparate_impact_threshold=1.1,
        )


def test_fairness_analyzer_validates_reference_group():
    """Explicit reference group must exist in the sensitive attribute."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "prediction": [1, 0, 1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="reference_group",
    ):
        FairnessAnalyzer(
            data,
            sensitive_attribute="group",
            prediction_column="prediction",
            reference_group="C",
        )


def test_group_metrics_calculate_selection_rates():
    """Group metrics should calculate representation and selection rates."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "prediction": [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )

    analyzer = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    )

    metrics = analyzer.group_metrics()

    assert metrics["A"]["count"] == 4
    assert metrics["A"]["positive_count"] == 3
    assert metrics["A"]["selection_rate"] == 0.75

    assert metrics["B"]["count"] == 4
    assert metrics["B"]["positive_count"] == 1
    assert metrics["B"]["selection_rate"] == 0.25

    assert metrics["A"]["representation_rate"] == 0.5
    assert metrics["B"]["representation_rate"] == 0.5


def test_highest_selection_rate_becomes_default_reference():
    """Highest-rate group should be reference when none is specified."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "prediction": [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )

    analyzer = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    )

    metrics = analyzer.group_metrics()

    assert analyzer.resolved_reference_group(metrics) == "A"


def test_explicit_reference_group_is_used():
    """Explicitly configured reference group should be preserved."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "prediction": [1, 1, 1, 0],
        }
    )

    analyzer = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
        reference_group="B",
    )

    metrics = analyzer.group_metrics()

    assert analyzer.resolved_reference_group(metrics) == "B"


def test_disparate_impact_indicator_flags_group_for_review():
    """Low disparate-impact ratio should produce a review indicator."""
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

    report = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    ).assess()

    assert report["reference_group"] == "A"

    assert (
        report["comparisons"]["A"]["disparate_impact_ratio"]
        == 1.0
    )

    assert (
        report["comparisons"]["B"]["disparate_impact_ratio"]
        == 0.5
    )

    assert report["comparisons"]["B"]["requires_review"] is True
    assert "B" in report["groups_requiring_review"]
    assert report["review_required"] is True


def test_groups_above_threshold_are_not_flagged():
    """Groups meeting the configured ratio should not require review."""
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

    report = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    ).assess()

    assert report["reference_group"] == "A"

    assert (
        report["comparisons"]["B"]["disparate_impact_ratio"]
        == 0.875
    )

    assert report["comparisons"]["B"]["requires_review"] is False
    assert report["groups_requiring_review"] == []
    assert report["review_required"] is False


def test_custom_positive_label_is_supported():
    """Analyzer should support non-numeric positive outcome labels."""
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

    report = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="decision",
        positive_label="approved",
    ).assess()

    assert report["positive_label"] == "approved"
    assert report["group_metrics"]["A"]["positive_count"] == 1
    assert report["group_metrics"]["B"]["positive_count"] == 2


def test_missing_values_are_excluded_and_reported():
    """Missing group and prediction values should be reported."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", None, "B"],
            "prediction": [1, None, 0, 1, 1],
        }
    )

    analyzer = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    )

    report = analyzer.assess()

    assert report["missing_data"]["missing_sensitive_attribute"] == 1
    assert report["missing_data"]["missing_predictions"] == 1
    assert report["missing_data"]["excluded_rows"] == 2

    assert report["group_metrics"]["A"]["count"] == 1
    assert report["group_metrics"]["B"]["count"] == 2


def test_selection_rate_difference_is_calculated():
    """Rate differences should be relative to the reference group."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "prediction": [
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
            ],
        }
    )

    report = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    ).assess()

    assert report["reference_group"] == "A"

    assert (
        report["comparisons"]["A"]["selection_rate_difference"]
        == 0.0
    )

    assert (
        report["comparisons"]["B"]["selection_rate_difference"]
        == -0.5
    )


def test_equal_zero_selection_rates_are_handled():
    """All-zero selection rates should not cause division errors."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "prediction": [0, 0, 0, 0],
        }
    )

    report = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    ).assess()

    for group in ["A", "B"]:
        assert (
            report["comparisons"][group]["disparate_impact_ratio"]
            == 1.0
        )
        assert (
            report["comparisons"][group]["requires_review"]
            is False
        )

    assert report["review_required"] is False


def test_assess_returns_expected_structure():
    """Fairness assessment should return the expected report structure."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "prediction": [1, 0, 1, 1],
        }
    )

    report = FairnessAnalyzer(
        data,
        sensitive_attribute="group",
        prediction_column="prediction",
    ).assess()

    expected_sections = {
        "sensitive_attribute",
        "prediction_column",
        "positive_label",
        "reference_group",
        "disparate_impact_threshold",
        "groups_analyzed",
        "group_metrics",
        "comparisons",
        "groups_requiring_review",
        "review_required",
        "missing_data",
    }

    assert set(report.keys()) == expected_sections
