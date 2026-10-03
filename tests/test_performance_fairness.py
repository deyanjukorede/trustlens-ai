import pandas as pd
import pytest

from trustlens.responsible_ai.performance_fairness import (
    PerformanceFairnessAnalyzer,
)


def test_analyzer_initialises_with_valid_data():
    """Analyzer should initialise with valid performance inputs."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "target": [1, 0, 1, 0],
            "prediction": [1, 0, 0, 0],
        }
    )

    analyzer = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
    )

    assert analyzer.sensitive_attribute == "group"
    assert analyzer.target_column == "target"
    assert analyzer.prediction_column == "prediction"
    assert analyzer.positive_label == 1
    assert analyzer.performance_gap_threshold == 0.10
    assert analyzer.minimum_group_size == 5


def test_analyzer_rejects_non_dataframe():
    """Analyzer should reject non-DataFrame input."""
    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
        PerformanceFairnessAnalyzer(
            {
                "group": ["A"],
                "target": [1],
                "prediction": [1],
            },
            sensitive_attribute="group",
            target_column="target",
            prediction_column="prediction",
        )


def test_analyzer_rejects_empty_dataframe():
    """Analyzer should reject an empty dataset."""
    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        PerformanceFairnessAnalyzer(
            pd.DataFrame(),
            sensitive_attribute="group",
            target_column="target",
            prediction_column="prediction",
        )


def test_analyzer_validates_sensitive_attribute():
    """Sensitive attribute must exist in the dataset."""
    data = pd.DataFrame(
        {
            "target": [1, 0],
            "prediction": [1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="sensitive_attribute",
    ):
        PerformanceFairnessAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            prediction_column="prediction",
        )


def test_analyzer_validates_target_column():
    """Target column must exist in the dataset."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "prediction": [1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="target_column",
    ):
        PerformanceFairnessAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            prediction_column="prediction",
        )


def test_analyzer_validates_prediction_column():
    """Prediction column must exist in the dataset."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "target": [1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="prediction_column",
    ):
        PerformanceFairnessAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            prediction_column="prediction",
        )


def test_analyzer_validates_performance_gap_threshold():
    """Performance-gap threshold must be between zero and one."""
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
        PerformanceFairnessAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            prediction_column="prediction",
            performance_gap_threshold=-0.01,
        )

    with pytest.raises(
        ValueError,
        match="performance_gap_threshold",
    ):
        PerformanceFairnessAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            prediction_column="prediction",
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

    analyzer = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        performance_gap_threshold=0,
    )

    assert analyzer.performance_gap_threshold == 0.0


def test_analyzer_validates_minimum_group_size():
    """Minimum group size should be a positive integer."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "target": [1, 0],
            "prediction": [1, 0],
        }
    )

    for value in [0, -1, 2.5, True]:
        with pytest.raises(
            ValueError,
            match="minimum_group_size",
        ):
            PerformanceFairnessAnalyzer(
                data,
                sensitive_attribute="group",
                target_column="target",
                prediction_column="prediction",
                minimum_group_size=value,
            )


def test_confusion_counts_are_calculated_correctly():
    """Binary confusion counts should be calculated correctly."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 8,
            "target": [1, 1, 1, 1, 0, 0, 0, 0],
            "prediction": [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )

    analyzer = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=1,
    )

    counts = analyzer.confusion_counts(data)

    assert counts == {
        "true_positive": 3,
        "true_negative": 3,
        "false_positive": 1,
        "false_negative": 1,
    }


def test_classification_metrics_are_calculated_correctly():
    """Performance metrics should match known confusion counts."""
    data = pd.DataFrame(
        {
            "group": ["A"],
            "target": [1],
            "prediction": [1],
        }
    )

    analyzer = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=1,
    )

    counts = {
        "true_positive": 3,
        "true_negative": 3,
        "false_positive": 1,
        "false_negative": 1,
    }

    metrics = analyzer.calculate_metrics(counts)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["error_rate"] == 0.25
    assert metrics["false_positive_rate"] == 0.25
    assert metrics["false_negative_rate"] == 0.25


def test_group_metrics_are_calculated_independently():
    """Each group should receive its own performance metrics."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "target": [
                1, 1, 0, 0,
                1, 1, 0, 0,
            ],
            "prediction": [
                1, 1, 0, 0,
                1, 0, 1, 0,
            ],
        }
    )

    analyzer = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=1,
    )

    metrics = analyzer.group_metrics()

    assert metrics["A"]["accuracy"] == 1.0
    assert metrics["A"]["error_rate"] == 0.0

    assert metrics["B"]["accuracy"] == 0.5
    assert metrics["B"]["error_rate"] == 0.5


def test_overall_metrics_are_calculated():
    """Analyzer should calculate performance across all valid rows."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "target": [
                1, 1, 0, 0,
                1, 1, 0, 0,
            ],
            "prediction": [
                1, 1, 0, 0,
                1, 0, 1, 0,
            ],
        }
    )

    analyzer = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=1,
    )

    overall = analyzer.overall_metrics()

    assert overall["count"] == 8
    assert overall["accuracy"] == 0.75
    assert overall["error_rate"] == 0.25


def test_performance_gap_is_detected():
    """Large group-to-overall performance gaps should be flagged."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "target": [
                1, 1, 0, 0,
                1, 1, 0, 0,
            ],
            "prediction": [
                1, 1, 0, 0,
                1, 0, 1, 0,
            ],
        }
    )

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        performance_gap_threshold=0.10,
        minimum_group_size=1,
    ).assess()

    assert report["overall_metrics"]["accuracy"] == 0.75

    assert (
        report["comparisons"]["A"]["metric_gaps"]["accuracy"]
        == 0.25
    )

    assert (
        report["comparisons"]["B"]["metric_gaps"]["accuracy"]
        == 0.25
    )

    assert (
        report["comparisons"]["A"]["performance_gap_detected"]
        is True
    )

    assert (
        report["comparisons"]["B"]["performance_gap_detected"]
        is True
    )

    assert report["review_required"] is True


def test_exact_threshold_gap_is_not_flagged():
    """A metric gap equal to the threshold should not be flagged."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "target": [1, 1, 1, 0, 0] * 2,
            "prediction": [
                1, 1, 1, 0, 0,
                1, 1, 0, 0, 0,
            ],
        }
    )

    analyzer = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        performance_gap_threshold=0.10,
        minimum_group_size=1,
    )

    metrics = analyzer.group_metrics()
    overall = analyzer.overall_metrics()
    comparisons = analyzer.performance_comparisons(
        metrics,
        overall,
    )

    assert metrics["A"]["accuracy"] == 1.0
    assert metrics["B"]["accuracy"] == 0.8
    assert overall["accuracy"] == 0.9

    assert comparisons["A"]["metric_gaps"]["accuracy"] == 0.1
    assert comparisons["B"]["metric_gaps"]["accuracy"] == 0.1

    assert (
        "accuracy"
        not in comparisons["A"]["flagged_metrics"]
    )

    assert (
        "accuracy"
        not in comparisons["B"]["flagged_metrics"]
    )


def test_limited_evidence_group_requires_review():
    """Groups below minimum size should be surfaced for review."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 2,
            "target": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "prediction": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=5,
    ).assess()

    assert report["group_metrics"]["A"]["limited_evidence"] is False
    assert report["group_metrics"]["B"]["limited_evidence"] is True

    assert (
        report["comparisons"]["B"]["requires_review"]
        is True
    )

    assert (
        report["review_summary"]["limited_evidence_groups"]
        == ["B"]
    )

    assert "B" in report["review_summary"]["groups_requiring_review"]
    assert report["review_required"] is True


def test_equal_group_performance_produces_no_review():
    """Equal performance with sufficient evidence should not flag review."""
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

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=5,
    ).assess()

    assert (
        report["review_summary"]["performance_gap_groups"]
        == []
    )

    assert (
        report["review_summary"]["limited_evidence_groups"]
        == []
    )

    assert (
        report["review_summary"]["groups_requiring_review"]
        == []
    )

    assert report["review_required"] is False


def test_custom_positive_label_is_supported():
    """Performance analysis should support custom positive labels."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A"],
            "target": [
                "approved",
                "approved",
                "denied",
                "denied",
            ],
            "prediction": [
                "approved",
                "denied",
                "approved",
                "denied",
            ],
        }
    )

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        positive_label="approved",
        minimum_group_size=1,
    ).assess()

    counts = report["group_metrics"]["A"]["confusion_counts"]

    assert report["positive_label"] == "approved"

    assert counts == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }

    assert report["group_metrics"]["A"]["accuracy"] == 0.5


def test_zero_denominators_are_handled():
    """Undefined binary rates should be handled without division errors."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "target": [0, 0, 0],
            "prediction": [0, 0, 0],
        }
    )

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=1,
    ).assess()

    metrics = report["group_metrics"]["A"]

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["error_rate"] == 0.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0


def test_missing_values_are_excluded_and_reported():
    """Missing analysis values should be excluded and reported."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", None, "B", "B"],
            "target": [1, None, 0, 1, 1, 0],
            "prediction": [1, 0, 0, 1, None, 0],
        }
    )

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=1,
    ).assess()

    assert report["missing_data"]["missing_sensitive_attribute"] == 1
    assert report["missing_data"]["missing_target"] == 1
    assert report["missing_data"]["missing_predictions"] == 1
    assert report["missing_data"]["excluded_rows"] == 3

    assert report["overall_metrics"]["count"] == 3
    assert report["group_metrics"]["A"]["count"] == 1
    assert report["group_metrics"]["B"]["count"] == 2


def test_all_invalid_rows_are_handled():
    """Analyzer should return empty group results if all rows are excluded."""
    data = pd.DataFrame(
        {
            "group": [None, None],
            "target": [1, None],
            "prediction": [None, 1],
        }
    )

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        minimum_group_size=1,
    ).assess()

    assert report["groups_analyzed"] == 0
    assert report["group_metrics"] == {}
    assert report["comparisons"] == {}

    assert report["overall_metrics"]["count"] == 0

    assert (
        report["review_summary"]["groups_requiring_review"]
        == []
    )

    assert report["review_required"] is False
    assert report["missing_data"]["excluded_rows"] == 2


def test_review_summary_identifies_performance_gap_groups():
    """Review summary should identify groups with metric disparities."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "target": [
                1, 1, 0, 0,
                1, 1, 0, 0,
            ],
            "prediction": [
                1, 1, 0, 0,
                1, 0, 1, 0,
            ],
        }
    )

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
        performance_gap_threshold=0.10,
        minimum_group_size=1,
    ).assess()

    assert "A" in report["review_summary"]["performance_gap_groups"]
    assert "B" in report["review_summary"]["performance_gap_groups"]

    assert "A" in report["review_summary"]["groups_requiring_review"]
    assert "B" in report["review_summary"]["groups_requiring_review"]


def test_assess_returns_expected_structure():
    """Performance assessment should expose expected report sections."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "target": [
                1, 1, 1, 0, 0,
                1, 1, 1, 0, 0,
            ],
            "prediction": [
                1, 1, 0, 0, 0,
                1, 0, 0, 0, 0,
            ],
        }
    )

    report = PerformanceFairnessAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        prediction_column="prediction",
    ).assess()

    expected_sections = {
        "sensitive_attribute",
        "target_column",
        "prediction_column",
        "positive_label",
        "performance_gap_threshold",
        "minimum_group_size",
        "groups_analyzed",
        "overall_metrics",
        "group_metrics",
        "comparisons",
        "review_summary",
        "review_required",
        "missing_data",
    }

    assert set(report.keys()) == expected_sections
