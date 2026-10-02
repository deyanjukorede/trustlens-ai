import pandas as pd
import pytest

from trustlens.readiness.feature_suitability import FeatureSuitabilityAnalyzer


@pytest.fixture
def sample_data():
    """Create a representative dataset for feature suitability testing."""
    return pd.DataFrame(
        {
            "age": [25, 31, 42, 36, 29],
            "income": [50000, 62000, 78000, 55000, 69000],
            "city": ["Sheffield", "Leeds", "London", "Sheffield", "Leeds"],
            "constant": ["active", "active", "active", "active", "active"],
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "target": [0, 1, 1, 0, 1],
        }
    )


def test_feature_columns_exclude_target(sample_data):
    """Target column should not be included in feature analysis."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
    )

    assert analyzer.feature_columns == [
        "age",
        "income",
        "city",
        "constant",
        "customer_id",
    ]


def test_feature_columns_without_target(sample_data):
    """All columns should be analysed when no target is supplied."""
    analyzer = FeatureSuitabilityAnalyzer(sample_data)

    assert analyzer.feature_columns == [
        "age",
        "income",
        "city",
        "constant",
        "customer_id",
        "target",
    ]


def test_constant_feature_detection(sample_data):
    """Constant features should be identified for review."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
    )

    result = analyzer.analyze_feature("constant")

    assert result["status"] == "review"
    assert "constant_feature" in result["issues"]
    assert result["unique_count"] == 1


def test_identifier_like_feature_detection(sample_data):
    """Highly unique features should be identified as identifier-like."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
    )

    result = analyzer.analyze_feature("customer_id")

    assert result["status"] == "review"
    assert "identifier_like" in result["issues"]
    assert result["unique_rate"] == 100.0


def test_high_cardinality_categorical_detection(sample_data):
    """High-cardinality categorical features should be identified."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
        high_cardinality_threshold=50.0,
    )

    result = analyzer.analyze_feature("city")

    assert result["status"] == "review"
    assert "high_cardinality" in result["issues"]
    assert result["unique_count"] == 3
    assert result["unique_rate"] == 60.0


def test_excessive_missingness_detection():
    """Features exceeding the missing threshold should be flagged."""
    data = pd.DataFrame(
        {
            "feature": [10, None, None, None, 50],
            "target": [0, 1, 0, 1, 0],
        }
    )

    analyzer = FeatureSuitabilityAnalyzer(
        data,
        target_column="target",
        missing_threshold=40.0,
    )

    result = analyzer.analyze_feature("feature")

    assert result["status"] == "review"
    assert "excessive_missingness" in result["issues"]
    assert result["missing_count"] == 3
    assert result["missing_rate"] == 60.0


def test_low_variation_detection():
    """Features dominated by one value should be flagged."""
    data = pd.DataFrame(
        {
            "feature": [1] * 19 + [2],
            "target": [0, 1] * 10,
        }
    )

    analyzer = FeatureSuitabilityAnalyzer(
        data,
        target_column="target",
        low_variation_threshold=95.0,
    )

    result = analyzer.analyze_feature("feature")

    assert result["status"] == "review"
    assert "low_variation" in result["issues"]
    assert result["dominant_value_rate"] == 95.0


def test_suitable_feature():
    """A feature without detected concerns should remain suitable."""
    data = pd.DataFrame(
        {
            "feature": [10, 10, 20, 20, 30, 30],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    analyzer = FeatureSuitabilityAnalyzer(
        data,
        target_column="target",
    )

    result = analyzer.analyze_feature("feature")

    assert result["status"] == "suitable"
    assert result["issues"] == []
    assert result["recommendations"] == []


def test_analyze_returns_expected_sections(sample_data):
    """Full analysis should expose expected summary sections."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
    )

    report = analyzer.analyze()

    assert "feature_count" in report
    assert "suitable_feature_count" in report
    assert "features_requiring_review_count" in report
    assert "suitability_rate" in report
    assert "suitable_features" in report
    assert "features_requiring_review" in report
    assert "issue_counts" in report
    assert "features" in report


def test_analyze_feature_count(sample_data):
    """Full analysis should report the correct number of features."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
    )

    report = analyzer.analyze()

    assert report["feature_count"] == 5
    assert "target" not in report["features"]


def test_issue_counts(sample_data):
    """Full analysis should summarise detected issue types."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
    )

    report = analyzer.analyze()

    assert report["issue_counts"]["constant_feature"] == 1
    assert report["issue_counts"]["identifier_like"] >= 1
    assert report["issue_counts"]["high_cardinality"] >= 1


def test_feature_metrics_are_reported(sample_data):
    """Per-feature analysis should expose structural metrics."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
    )

    result = analyzer.analyze_feature("city")

    assert "data_type" in result
    assert "missing_count" in result
    assert "missing_rate" in result
    assert "non_missing_count" in result
    assert "unique_count" in result
    assert "unique_rate" in result
    assert "dominant_value_rate" in result


def test_invalid_input():
    """Analyzer should reject input that is not a pandas DataFrame."""
    with pytest.raises(TypeError):
        FeatureSuitabilityAnalyzer([1, 2, 3])


def test_empty_dataframe():
    """Analyzer should reject an empty DataFrame."""
    with pytest.raises(ValueError):
        FeatureSuitabilityAnalyzer(pd.DataFrame())


def test_invalid_target_column(sample_data):
    """Analyzer should reject a target column that does not exist."""
    with pytest.raises(ValueError):
        FeatureSuitabilityAnalyzer(
            sample_data,
            target_column="nonexistent_target",
        )


@pytest.mark.parametrize(
    "threshold_name",
    [
        "missing_threshold",
        "high_cardinality_threshold",
        "identifier_threshold",
        "low_variation_threshold",
    ],
)
def test_threshold_below_zero_is_rejected(sample_data, threshold_name):
    """Percentage thresholds below zero should be rejected."""
    arguments = {
        threshold_name: -1,
    }

    with pytest.raises(ValueError):
        FeatureSuitabilityAnalyzer(
            sample_data,
            **arguments,
        )


@pytest.mark.parametrize(
    "threshold_name",
    [
        "missing_threshold",
        "high_cardinality_threshold",
        "identifier_threshold",
        "low_variation_threshold",
    ],
)
def test_threshold_above_100_is_rejected(sample_data, threshold_name):
    """Percentage thresholds above 100 should be rejected."""
    arguments = {
        threshold_name: 101,
    }

    with pytest.raises(ValueError):
        FeatureSuitabilityAnalyzer(
            sample_data,
            **arguments,
        )


def test_non_numeric_threshold_is_rejected(sample_data):
    """Thresholds must be numeric."""
    with pytest.raises(TypeError):
        FeatureSuitabilityAnalyzer(
            sample_data,
            missing_threshold="high",
        )


def test_target_cannot_be_analyzed_as_feature(sample_data):
    """Explicit feature analysis should reject the target column."""
    analyzer = FeatureSuitabilityAnalyzer(
        sample_data,
        target_column="target",
    )

    with pytest.raises(ValueError):
        analyzer.analyze_feature("target")


def test_missing_only_feature():
    """A completely missing feature should be identified for review."""
    data = pd.DataFrame(
        {
            "empty_feature": [None, None, None, None],
            "target": [0, 1, 0, 1],
        }
    )

    analyzer = FeatureSuitabilityAnalyzer(
        data,
        target_column="target",
    )

    result = analyzer.analyze_feature("empty_feature")

    assert result["status"] == "review"
    assert "constant_feature" in result["issues"]
    assert "excessive_missingness" in result["issues"]
    assert result["missing_rate"] == 100.0
    assert result["non_missing_count"] == 0


def test_custom_identifier_threshold():
    """Custom identifier thresholds should affect detection."""
    data = pd.DataFrame(
        {
            "reference": ["A", "B", "C", "D", "D"],
            "target": [0, 1, 0, 1, 0],
        }
    )

    analyzer = FeatureSuitabilityAnalyzer(
        data,
        target_column="target",
        identifier_threshold=80.0,
    )

    result = analyzer.analyze_feature("reference")

    assert "identifier_like" in result["issues"]


def test_suitability_rate():
    """Suitability rate should reflect suitable and review features."""
    data = pd.DataFrame(
        {
            "good_feature": [10, 10, 20, 20, 30, 30],
            "constant_feature": [1, 1, 1, 1, 1, 1],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    analyzer = FeatureSuitabilityAnalyzer(
        data,
        target_column="target",
    )

    report = analyzer.analyze()

    assert report["feature_count"] == 2
    assert report["suitable_feature_count"] == 1
    assert report["features_requiring_review_count"] == 1
    assert report["suitability_rate"] == 50.0
