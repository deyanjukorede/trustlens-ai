import pandas as pd
import pytest

from trustlens.operational_trust.stability import DataStabilityAnalyzer


def test_analyzer_initialises_with_valid_dataframe():
    """Analyzer should initialise with valid data and defaults."""
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": ["A", "B", "C"],
        }
    )

    analyzer = DataStabilityAnalyzer(data)

    assert analyzer.row_count == 3
    assert analyzer.column_count == 2
    assert analyzer.high_missingness_threshold == 0.50
    assert analyzer.near_constant_threshold == 0.95
    assert analyzer.high_cardinality_threshold == 0.90
    assert analyzer.duplicate_rate_threshold == 0.10


def test_analyzer_rejects_non_dataframe_input():
    """Data must be supplied as a pandas DataFrame."""
    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
        DataStabilityAnalyzer(
            {
                "feature": [1, 2, 3],
            }
        )


def test_analyzer_rejects_empty_dataframe():
    """Data must not be empty."""
    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        DataStabilityAnalyzer(pd.DataFrame())


@pytest.mark.parametrize(
    "threshold_name",
    [
        "high_missingness_threshold",
        "near_constant_threshold",
        "high_cardinality_threshold",
        "duplicate_rate_threshold",
    ],
)
def test_thresholds_reject_non_numeric_values(threshold_name):
    """All stability thresholds should require numeric values."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    kwargs = {
        threshold_name: "invalid",
    }

    with pytest.raises(
        TypeError,
        match=f"{threshold_name} must be numeric",
    ):
        DataStabilityAnalyzer(
            data,
            **kwargs,
        )


@pytest.mark.parametrize(
    "threshold_name",
    [
        "high_missingness_threshold",
        "near_constant_threshold",
        "high_cardinality_threshold",
        "duplicate_rate_threshold",
    ],
)
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_thresholds_reject_values_outside_zero_and_one(
    threshold_name,
    value,
):
    """Proportion thresholds should remain between zero and one."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    kwargs = {
        threshold_name: value,
    }

    with pytest.raises(
        ValueError,
        match=f"{threshold_name} must be between 0 and 1",
    ):
        DataStabilityAnalyzer(
            data,
            **kwargs,
        )


def test_zero_and_one_threshold_boundaries_are_allowed():
    """Zero and one should be valid threshold boundaries."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    zero = DataStabilityAnalyzer(
        data,
        high_missingness_threshold=0,
        near_constant_threshold=0,
        high_cardinality_threshold=0,
        duplicate_rate_threshold=0,
    )

    one = DataStabilityAnalyzer(
        data,
        high_missingness_threshold=1,
        near_constant_threshold=1,
        high_cardinality_threshold=1,
        duplicate_rate_threshold=1,
    )

    assert zero.high_missingness_threshold == 0.0
    assert zero.near_constant_threshold == 0.0
    assert zero.high_cardinality_threshold == 0.0
    assert zero.duplicate_rate_threshold == 0.0

    assert one.high_missingness_threshold == 1.0
    assert one.near_constant_threshold == 1.0
    assert one.high_cardinality_threshold == 1.0
    assert one.duplicate_rate_threshold == 1.0


def test_high_missingness_feature_requires_review():
    """Missingness above the configured threshold should be flagged."""
    data = pd.DataFrame(
        {
            "feature": [1.0, None, None, None],
            "other": [1, 2, 3, 4],
        }
    )

    result = DataStabilityAnalyzer(
        data,
        high_missingness_threshold=0.50,
    ).assess()

    missingness = result["feature_analysis"]["feature"]["missingness"]

    assert missingness["missing_count"] == 3
    assert missingness["missing_rate"] == 0.75
    assert missingness["requires_review"] is True

    assert result["high_missingness_features"] == ["feature"]
    assert "high_missingness" in result["review_reasons"]
    assert result["review_required"] is True


def test_missingness_exactly_on_threshold_is_not_flagged():
    """Missingness equal to the threshold should not require review."""
    data = pd.DataFrame(
        {
            "feature": [1.0, 2.0, None, None],
        }
    )

    result = DataStabilityAnalyzer(
        data,
        high_missingness_threshold=0.50,
    ).assess()

    missingness = result["feature_analysis"]["feature"]["missingness"]

    assert missingness["missing_rate"] == 0.50
    assert missingness["requires_review"] is False
    assert result["high_missingness_features"] == []


def test_constant_feature_requires_review():
    """A constant non-missing feature should be surfaced."""
    data = pd.DataFrame(
        {
            "constant": [7, 7, 7, 7],
            "variable": [1, 2, 3, 4],
        }
    )

    result = DataStabilityAnalyzer(data).assess()

    concentration = (
        result["feature_analysis"]["constant"]["concentration"]
    )

    assert concentration["unique_non_missing_values"] == 1
    assert concentration["dominant_value_rate"] == 1.0
    assert concentration["is_constant"] is True
    assert concentration["is_near_constant"] is False
    assert concentration["requires_review"] is True

    assert result["constant_features"] == ["constant"]
    assert "constant_features" in result["review_reasons"]


def test_all_missing_feature_is_not_classified_as_constant():
    """All-missing data should be represented through missingness."""
    data = pd.DataFrame(
        {
            "feature": [None, None, None, None],
            "other": [1, 2, 3, 4],
        }
    )

    result = DataStabilityAnalyzer(data).assess()

    concentration = (
        result["feature_analysis"]["feature"]["concentration"]
    )

    assert concentration["unique_non_missing_values"] == 0
    assert concentration["dominant_value_rate"] is None
    assert concentration["is_constant"] is False
    assert concentration["is_near_constant"] is False

    assert "feature" in result["high_missingness_features"]
    assert "feature" not in result["constant_features"]


def test_near_constant_feature_requires_review():
    """Dominant values above the threshold should be surfaced."""
    data = pd.DataFrame(
        {
            "feature": ["A"] * 19 + ["B"],
        }
    )

    result = DataStabilityAnalyzer(
        data,
        near_constant_threshold=0.90,
    ).assess()

    concentration = (
        result["feature_analysis"]["feature"]["concentration"]
    )

    assert concentration["dominant_value_rate"] == 0.95
    assert concentration["is_constant"] is False
    assert concentration["is_near_constant"] is True
    assert concentration["requires_review"] is True

    assert result["near_constant_features"] == ["feature"]
    assert "near_constant_features" in result["review_reasons"]


def test_near_constant_exactly_on_threshold_is_not_flagged():
    """Dominant rate equal to the threshold should not be flagged."""
    data = pd.DataFrame(
        {
            "feature": ["A"] * 9 + ["B"],
        }
    )

    result = DataStabilityAnalyzer(
        data,
        near_constant_threshold=0.90,
    ).assess()

    concentration = (
        result["feature_analysis"]["feature"]["concentration"]
    )

    assert concentration["dominant_value_rate"] == 0.90
    assert concentration["is_near_constant"] is False


def test_high_cardinality_categorical_feature_requires_review():
    """High categorical uniqueness should be surfaced."""
    data = pd.DataFrame(
        {
            "customer_code": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C005",
                "C006",
                "C007",
                "C008",
                "C009",
                "C009",
            ]
        }
    )

    result = DataStabilityAnalyzer(
        data,
        high_cardinality_threshold=0.80,
    ).assess()

    cardinality = (
        result["feature_analysis"]["customer_code"]["cardinality"]
    )

    assert cardinality["unique_count"] == 9
    assert cardinality["unique_rate"] == 0.9
    assert cardinality["is_categorical"] is True
    assert cardinality["high_cardinality"] is True
    assert cardinality["requires_review"] is True

    assert result["high_cardinality_features"] == [
        "customer_code"
    ]

    assert "high_cardinality" in result["review_reasons"]


def test_numeric_uniqueness_is_not_high_cardinality_indicator():
    """Unique numeric measurements should not be flagged as categorical."""
    data = pd.DataFrame(
        {
            "measurement": [
                10.1,
                11.2,
                12.3,
                13.4,
                14.5,
            ]
        }
    )

    result = DataStabilityAnalyzer(data).assess()

    cardinality = (
        result["feature_analysis"]["measurement"]["cardinality"]
    )

    assert cardinality["unique_rate"] == 1.0
    assert cardinality["is_categorical"] is False
    assert cardinality["high_cardinality"] is False
    assert cardinality["requires_review"] is False


def test_datetime_uniqueness_is_not_high_cardinality_indicator():
    """Datetime uniqueness should not be treated as categorical."""
    data = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                ]
            )
        }
    )

    result = DataStabilityAnalyzer(data).assess()

    cardinality = (
        result["feature_analysis"]["event_time"]["cardinality"]
    )

    assert cardinality["is_categorical"] is False
    assert cardinality["high_cardinality"] is False


def test_duplicate_pressure_above_threshold_requires_review():
    """Duplicate-row pressure above the threshold should be flagged."""
    data = pd.DataFrame(
        {
            "feature": [1, 1, 1, 2, 3],
            "segment": ["A", "A", "A", "B", "C"],
        }
    )

    result = DataStabilityAnalyzer(
        data,
        duplicate_rate_threshold=0.20,
    ).assess()

    duplicates = result["duplicate_analysis"]

    assert duplicates["duplicate_rows"] == 2
    assert duplicates["duplicate_rate"] == 0.4
    assert duplicates["requires_review"] is True

    assert "duplicate_pressure" in result["review_reasons"]
    assert result["review_required"] is True


def test_duplicate_rate_exactly_on_threshold_is_not_flagged():
    """Duplicate rate equal to threshold should not require review."""
    data = pd.DataFrame(
        {
            "feature": [1, 1, 2, 3],
        }
    )

    result = DataStabilityAnalyzer(
        data,
        duplicate_rate_threshold=0.25,
    ).assess()

    duplicates = result["duplicate_analysis"]

    assert duplicates["duplicate_rows"] == 1
    assert duplicates["duplicate_rate"] == 0.25
    assert duplicates["requires_review"] is False


def test_feature_type_summary_counts_supported_types():
    """Dataset structure should summarise broad feature types."""
    data = pd.DataFrame(
        {
            "numeric": [1, 2, 3],
            "category": ["A", "B", "C"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
        }
    )

    result = DataStabilityAnalyzer(data).assess()

    assert result["feature_type_summary"] == {
        "numeric_features": 1,
        "categorical_features": 1,
        "datetime_features": 1,
    }


def test_stable_dataset_produces_no_review_indicators():
    """Dataset without configured stability indicators should be clear."""
    data = pd.DataFrame(
        {
            "score": [10, 20, 30, 40],
            "segment": ["A", "B", "A", "B"],
        }
    )

    result = DataStabilityAnalyzer(data).assess()

    assert result["high_missingness_features"] == []
    assert result["constant_features"] == []
    assert result["near_constant_features"] == []
    assert result["high_cardinality_features"] == []
    assert result["duplicate_analysis"]["requires_review"] is False
    assert result["review_reasons"] == []
    assert result["review_required"] is False


def test_multiple_stability_indicators_can_coexist():
    """Multiple measurable stability indicators should be preserved."""
    data = pd.DataFrame(
        {
            "mostly_missing": [
                1.0,
                None,
                None,
                None,
                None,
            ],
            "constant": [7, 7, 7, 7, 7],
        }
    )

    result = DataStabilityAnalyzer(
        data,
        high_missingness_threshold=0.50,
    ).assess()

    assert result["high_missingness_features"] == [
        "mostly_missing"
    ]

    assert "constant" in result["constant_features"]

    assert "high_missingness" in result["review_reasons"]
    assert "constant_features" in result["review_reasons"]
    assert result["review_required"] is True


def test_assessment_returns_expected_top_level_structure():
    """Stability assessment should expose a stable output contract."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
        }
    )

    result = DataStabilityAnalyzer(data).assess()

    assert set(result.keys()) == {
        "dataset",
        "thresholds",
        "feature_type_summary",
        "feature_analysis",
        "high_missingness_features",
        "constant_features",
        "near_constant_features",
        "high_cardinality_features",
        "duplicate_analysis",
        "review_required",
        "review_reasons",
    }
