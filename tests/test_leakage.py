import pandas as pd
import pytest

from trustlens.readiness.leakage import LeakageRiskAnalyzer


@pytest.fixture
def safe_data():
    """Create a dataset without obvious structural leakage."""
    return pd.DataFrame(
        {
            "age": [20, 45, 31, 58, 27, 49],
            "income": [32000, 71000, 48000, 83000, 39000, 65000],
            "city": [
                "Sheffield",
                "London",
                "Leeds",
                "Manchester",
                "Sheffield",
                "London",
            ],
            "target": [0, 1, 1, 0, 0, 1],
        }
    )


def test_feature_columns_exclude_target(safe_data):
    """Target column should not be treated as a modelling feature."""
    analyzer = LeakageRiskAnalyzer(
        safe_data,
        target_column="target",
    )

    assert analyzer.feature_columns == [
        "age",
        "income",
        "city",
    ]


def test_duplicate_target_detection():
    """Exact copies of the target should be detected."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "target_copy": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    assert analyzer.duplicate_target_features() == [
        "target_copy"
    ]


def test_duplicate_target_creates_high_risk():
    """An exact target duplicate should produce high leakage risk."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "target_copy": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    assert analyzer.risk_level() == "high"


def test_high_correlation_detection():
    """Highly correlated numeric features should be flagged."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40, 50],
            "target": [1, 2, 3, 4, 5],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    correlations = analyzer.high_correlation_features()

    assert "feature" in correlations
    assert correlations["feature"] == 1.0


def test_non_numeric_target_has_no_correlation_analysis():
    """Correlation analysis should be skipped for non-numeric targets."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": ["yes", "no", "yes", "no"],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    assert analyzer.high_correlation_features() == {}


def test_identifier_like_feature_detection():
    """Nearly unique columns should be identified as identifier-like."""
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C005",
            ],
            "group": ["A", "A", "B", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    identifiers = analyzer.identifier_like_features()

    assert "customer_id" in identifiers
    assert identifiers["customer_id"] == 1.0


def test_low_cardinality_feature_not_identifier():
    """Repeated categorical values should not be identifier-like."""
    data = pd.DataFrame(
        {
            "city": [
                "Sheffield",
                "London",
                "Sheffield",
                "London",
                "Sheffield",
            ],
            "target": [0, 1, 0, 1, 0],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    assert "city" not in analyzer.identifier_like_features()


def test_suspicious_feature_name():
    """Target-derived feature names should generate warnings."""
    data = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "predicted_target": [0.1, 0.8, 0.2, 0.9],
            "target": [0, 1, 0, 1],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    assert "predicted_target" in (
        analyzer.suspicious_name_features()
    )


def test_risk_features_are_unique():
    """Combined risk feature output should contain unique columns."""
    data = pd.DataFrame(
        {
            "target_copy": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    risk_features = analyzer.risk_features()

    assert risk_features.count("target_copy") == 1


def test_assess_returns_expected_sections():
    """Full assessment should expose leakage-risk sections."""
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "feature": [5, 7, 2, 9],
            "target": [0, 1, 0, 1],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    report = analyzer.assess()

    assert report["target_column"] == "target"
    assert report["features_analyzed"] == 2
    assert "duplicate_target_features" in report
    assert "high_correlation_features" in report
    assert "identifier_like_features" in report
    assert "suspicious_name_features" in report
    assert "risk_features" in report
    assert "risk_feature_count" in report
    assert "risk_level" in report
    assert "correlation_threshold" in report
    assert "high_cardinality_threshold" in report


def test_low_risk_when_no_warning_signals():
    """Dataset without structural warnings should report low risk."""
    data = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B"],
            "region": ["N", "N", "S", "S", "N", "S"],
            "target": ["yes", "no", "yes", "no", "no", "yes"],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    assert analyzer.risk_level() == "low"
    assert analyzer.risk_features() == []


def test_medium_risk_for_identifier():
    """Identifier-like features should produce medium risk."""
    data = pd.DataFrame(
        {
            "customer_id": [
                "A001",
                "A002",
                "A003",
                "A004",
            ],
            "target": ["yes", "no", "yes", "no"],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
    )

    assert analyzer.risk_level() == "medium"


def test_custom_correlation_threshold():
    """Custom correlation thresholds should be respected."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [1, 2, 3, 5, 4],
        }
    )

    analyzer = LeakageRiskAnalyzer(
        data,
        target_column="target",
        correlation_threshold=0.80,
    )

    assert "feature" in analyzer.high_correlation_features()


def test_invalid_input():
    """Analyzer should reject non-DataFrame input."""
    with pytest.raises(TypeError):
        LeakageRiskAnalyzer(
            [1, 2, 3],
            target_column="target",
        )


def test_empty_dataframe():
    """Analyzer should reject an empty DataFrame."""
    with pytest.raises(ValueError):
        LeakageRiskAnalyzer(
            pd.DataFrame(),
            target_column="target",
        )


def test_invalid_target_column(safe_data):
    """Analyzer should reject a missing target column."""
    with pytest.raises(ValueError):
        LeakageRiskAnalyzer(
            safe_data,
            target_column="missing_target",
        )


@pytest.mark.parametrize(
    "threshold",
    [0, -0.1, 1.1],
)
def test_invalid_correlation_threshold(safe_data, threshold):
    """Invalid correlation thresholds should be rejected."""
    with pytest.raises(ValueError):
        LeakageRiskAnalyzer(
            safe_data,
            target_column="target",
            correlation_threshold=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    [0, -0.1, 1.1],
)
def test_invalid_cardinality_threshold(safe_data, threshold):
    """Invalid cardinality thresholds should be rejected."""
    with pytest.raises(ValueError):
        LeakageRiskAnalyzer(
            safe_data,
            target_column="target",
            high_cardinality_threshold=threshold,
        )
