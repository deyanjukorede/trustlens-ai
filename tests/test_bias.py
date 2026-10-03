import pandas as pd
import pytest

from trustlens.responsible_ai.bias import BiasIndicatorAnalyzer


def test_bias_analyzer_initialises_with_valid_data():
    """Analyzer should initialise with valid bias-analysis inputs."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "target": [1, 0, 1, 1],
        }
    )

    analyzer = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
    )

    assert analyzer.sensitive_attribute == "group"
    assert analyzer.target_column == "target"
    assert analyzer.positive_label == 1
    assert analyzer.representation_threshold == 0.10
    assert analyzer.outcome_ratio_threshold == 0.80
    assert analyzer.minimum_group_size == 5


def test_bias_analyzer_rejects_non_dataframe():
    """Analyzer should reject non-DataFrame input."""
    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
        BiasIndicatorAnalyzer(
            {"group": ["A"], "target": [1]},
            sensitive_attribute="group",
            target_column="target",
        )


def test_bias_analyzer_rejects_empty_dataframe():
    """Analyzer should reject an empty dataset."""
    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        BiasIndicatorAnalyzer(
            pd.DataFrame(),
            sensitive_attribute="group",
            target_column="target",
        )


def test_bias_analyzer_validates_sensitive_attribute():
    """Sensitive attribute must exist in the dataset."""
    data = pd.DataFrame(
        {
            "target": [1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="sensitive_attribute",
    ):
        BiasIndicatorAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
        )


def test_bias_analyzer_validates_target_column():
    """Target column must exist in the dataset."""
    data = pd.DataFrame(
        {
            "group": ["A", "B", "A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="target_column",
    ):
        BiasIndicatorAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
        )


def test_bias_analyzer_validates_representation_threshold():
    """Representation threshold should be within the valid range."""
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
        BiasIndicatorAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            representation_threshold=0,
        )

    with pytest.raises(
        ValueError,
        match="representation_threshold",
    ):
        BiasIndicatorAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            representation_threshold=1.1,
        )


def test_bias_analyzer_validates_outcome_ratio_threshold():
    """Outcome-ratio threshold should be within the valid range."""
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
        BiasIndicatorAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            outcome_ratio_threshold=0,
        )

    with pytest.raises(
        ValueError,
        match="outcome_ratio_threshold",
    ):
        BiasIndicatorAnalyzer(
            data,
            sensitive_attribute="group",
            target_column="target",
            outcome_ratio_threshold=1.1,
        )


def test_bias_analyzer_validates_minimum_group_size():
    """Minimum group size should be a positive integer."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "target": [1, 0],
        }
    )

    invalid_values = [0, -1, 2.5, True]

    for value in invalid_values:
        with pytest.raises(
            ValueError,
            match="minimum_group_size",
        ):
            BiasIndicatorAnalyzer(
                data,
                sensitive_attribute="group",
                target_column="target",
                minimum_group_size=value,
            )


def test_group_metrics_calculate_representation():
    """Group metrics should calculate representation correctly."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 2,
            "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    analyzer = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        minimum_group_size=1,
    )

    metrics = analyzer.group_metrics()

    assert metrics["A"]["count"] == 8
    assert metrics["A"]["representation_rate"] == 0.8

    assert metrics["B"]["count"] == 2
    assert metrics["B"]["representation_rate"] == 0.2


def test_underrepresentation_indicator():
    """Groups below the representation threshold should be flagged."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 9 + ["B"],
            "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        representation_threshold=0.20,
        minimum_group_size=1,
    ).assess()

    assert report["group_metrics"]["A"]["underrepresented"] is False
    assert report["group_metrics"]["B"]["underrepresented"] is True

    assert (
        report["indicators"]["underrepresented_groups"]
        == ["B"]
    )

    assert (
        "underrepresentation"
        in report["indicators"]["indicators_by_group"]["B"]
    )


def test_small_group_indicator():
    """Groups below the minimum sample size should be flagged."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 2,
            "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        minimum_group_size=5,
    ).assess()

    assert report["group_metrics"]["A"]["small_group"] is False
    assert report["group_metrics"]["B"]["small_group"] is True

    assert report["indicators"]["small_groups"] == ["B"]

    assert (
        "small_group"
        in report["indicators"]["indicators_by_group"]["B"]
    )


def test_highest_outcome_rate_becomes_reference_group():
    """Highest observed outcome-rate group should become reference."""
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

    analyzer = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
    )

    metrics = analyzer.group_metrics()

    assert analyzer.reference_group(metrics) == "A"


def test_outcome_disparity_indicator():
    """Large observed outcome disparity should produce an indicator."""
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

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
    ).assess()

    comparison = report["outcome_comparisons"]["B"]

    assert comparison["positive_outcome_rate"] == 0.4
    assert comparison["reference_outcome_rate"] == 0.8
    assert comparison["outcome_rate_ratio"] == 0.5
    assert comparison["outcome_rate_difference"] == -0.4
    assert comparison["outcome_disparity"] is True

    assert (
        report["indicators"]["outcome_disparity_groups"]
        == ["B"]
    )


def test_group_can_have_multiple_bias_indicators():
    """A group may trigger more than one review indicator."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 19 + ["B"],
            "target": (
                [1] * 15
                + [0] * 4
                + [0]
            ),
        }
    )

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        representation_threshold=0.10,
        minimum_group_size=5,
    ).assess()

    indicators = report["indicators"]["indicators_by_group"]["B"]

    assert "underrepresentation" in indicators
    assert "small_group" in indicators
    assert "outcome_disparity" in indicators

    assert report["review_required"] is True


def test_custom_positive_label_is_supported():
    """Analyzer should support non-numeric observed outcomes."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "outcome": [
                "approved",
                "denied",
                "approved",
                "approved",
            ],
        }
    )

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="outcome",
        positive_label="approved",
        minimum_group_size=1,
    ).assess()

    assert report["positive_label"] == "approved"
    assert report["group_metrics"]["A"]["positive_count"] == 1
    assert report["group_metrics"]["B"]["positive_count"] == 2


def test_missing_values_are_excluded_and_reported():
    """Missing group and target values should be reported."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", None, "B"],
            "target": [1, None, 0, 1, 1],
        }
    )

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        minimum_group_size=1,
    ).assess()

    assert report["missing_data"]["missing_sensitive_attribute"] == 1
    assert report["missing_data"]["missing_target"] == 1
    assert report["missing_data"]["excluded_rows"] == 2

    assert report["group_metrics"]["A"]["count"] == 1
    assert report["group_metrics"]["B"]["count"] == 2


def test_equal_zero_outcome_rates_are_handled():
    """All-zero outcome rates should not cause division errors."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "target": [0] * 10,
        }
    )

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        minimum_group_size=1,
    ).assess()

    for group in ["A", "B"]:
        assert (
            report["outcome_comparisons"][group][
                "outcome_rate_ratio"
            ]
            == 1.0
        )

        assert (
            report["outcome_comparisons"][group][
                "outcome_disparity"
            ]
            is False
        )


def test_no_indicators_produces_no_review():
    """Dataset without configured indicators should not require review."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "target": (
                [1] * 8
                + [0] * 2
                + [1] * 7
                + [0] * 3
            ),
        }
    )

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
        minimum_group_size=5,
    ).assess()

    assert report["indicators"]["underrepresented_groups"] == []
    assert report["indicators"]["small_groups"] == []
    assert report["indicators"]["outcome_disparity_groups"] == []
    assert report["indicators"]["groups_with_indicators"] == []
    assert report["review_required"] is False


def test_assess_returns_expected_structure():
    """Bias assessment should expose the expected report structure."""
    data = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "target": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        }
    )

    report = BiasIndicatorAnalyzer(
        data,
        sensitive_attribute="group",
        target_column="target",
    ).assess()

    expected_sections = {
        "sensitive_attribute",
        "target_column",
        "positive_label",
        "representation_threshold",
        "outcome_ratio_threshold",
        "minimum_group_size",
        "groups_analyzed",
        "reference_group",
        "group_metrics",
        "outcome_comparisons",
        "indicators",
        "review_required",
        "missing_data",
    }

    assert set(report.keys()) == expected_sections
