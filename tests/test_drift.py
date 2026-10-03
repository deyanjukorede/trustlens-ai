import pandas as pd
import pytest

from trustlens.operational_trust.drift import DataDriftAnalyzer


def test_analyzer_initialises_with_valid_dataframes():
    """Analyzer should accept valid reference and current datasets."""
    reference = pd.DataFrame({"feature": [1, 2, 3]})
    current = pd.DataFrame({"feature": [2, 3, 4]})

    analyzer = DataDriftAnalyzer(reference, current)

    assert analyzer.numeric_mean_shift_threshold == 0.20
    assert analyzer.categorical_distribution_threshold == 0.20


def test_analyzer_rejects_non_dataframe_reference_data():
    """Reference data must be a pandas DataFrame."""
    current = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match="reference_data must be a pandas DataFrame",
    ):
        DataDriftAnalyzer(
            {"feature": [1, 2, 3]},
            current,
        )


def test_analyzer_rejects_non_dataframe_current_data():
    """Current data must be a pandas DataFrame."""
    reference = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match="current_data must be a pandas DataFrame",
    ):
        DataDriftAnalyzer(
            reference,
            {"feature": [1, 2, 3]},
        )


def test_analyzer_rejects_empty_reference_data():
    """Reference data must not be empty."""
    current = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="reference_data must not be empty",
    ):
        DataDriftAnalyzer(pd.DataFrame(), current)


def test_analyzer_rejects_empty_current_data():
    """Current data must not be empty."""
    reference = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="current_data must not be empty",
    ):
        DataDriftAnalyzer(reference, pd.DataFrame())


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_numeric_threshold_is_rejected(value):
    """Numeric drift threshold should remain between zero and one."""
    reference = pd.DataFrame({"feature": [1, 2, 3]})
    current = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="numeric_mean_shift_threshold",
    ):
        DataDriftAnalyzer(
            reference,
            current,
            numeric_mean_shift_threshold=value,
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_categorical_threshold_is_rejected(value):
    """Categorical drift threshold should remain between zero and one."""
    reference = pd.DataFrame({"feature": ["A", "B"]})
    current = pd.DataFrame({"feature": ["A", "B"]})

    with pytest.raises(
        ValueError,
        match="categorical_distribution_threshold",
    ):
        DataDriftAnalyzer(
            reference,
            current,
            categorical_distribution_threshold=value,
        )


def test_threshold_boundaries_are_allowed():
    """Zero and one should be valid drift thresholds."""
    reference = pd.DataFrame({"feature": [1, 2, 3]})
    current = pd.DataFrame({"feature": [1, 2, 3]})

    zero = DataDriftAnalyzer(
        reference,
        current,
        numeric_mean_shift_threshold=0,
        categorical_distribution_threshold=0,
    )

    one = DataDriftAnalyzer(
        reference,
        current,
        numeric_mean_shift_threshold=1,
        categorical_distribution_threshold=1,
    )

    assert zero.numeric_mean_shift_threshold == 0.0
    assert zero.categorical_distribution_threshold == 0.0
    assert one.numeric_mean_shift_threshold == 1.0
    assert one.categorical_distribution_threshold == 1.0


def test_column_alignment_reports_schema_changes():
    """Schema alignment should identify shared and unique columns."""
    reference = pd.DataFrame(
        {
            "shared": [1, 2],
            "removed": [3, 4],
        }
    )

    current = pd.DataFrame(
        {
            "shared": [1, 2],
            "added": [5, 6],
        }
    )

    alignment = DataDriftAnalyzer(
        reference,
        current,
    ).column_alignment()

    assert alignment["shared_columns"] == ["shared"]
    assert alignment["reference_only_columns"] == ["removed"]
    assert alignment["current_only_columns"] == ["added"]


def test_identical_numeric_feature_has_no_drift_indicator():
    """Identical numeric distributions should not require review."""
    reference = pd.DataFrame(
        {"feature": [10, 20, 30, 40]}
    )

    current = pd.DataFrame(
        {"feature": [10, 20, 30, 40]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    analysis = result["feature_analysis"]["feature"]["drift_analysis"]

    assert analysis["reference_mean"] == 25.0
    assert analysis["current_mean"] == 25.0
    assert analysis["relative_mean_shift"] == 0.0
    assert analysis["requires_review"] is False
    assert result["features_requiring_review"] == []
    assert result["review_required"] is False


def test_numeric_mean_shift_above_threshold_requires_review():
    """A numeric mean shift above the threshold should be flagged."""
    reference = pd.DataFrame(
        {"feature": [10, 10, 10, 10]}
    )

    current = pd.DataFrame(
        {"feature": [15, 15, 15, 15]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
        numeric_mean_shift_threshold=0.20,
    ).assess()

    analysis = result["feature_analysis"]["feature"]["drift_analysis"]

    assert analysis["relative_mean_shift"] == 0.5
    assert analysis["requires_review"] is True
    assert result["features_requiring_review"] == ["feature"]
    assert result["review_required"] is True


def test_numeric_shift_exactly_on_threshold_is_not_flagged():
    """A numeric shift equal to the threshold should not be flagged."""
    reference = pd.DataFrame(
        {"feature": [10, 10, 10, 10]}
    )

    current = pd.DataFrame(
        {"feature": [12, 12, 12, 12]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
        numeric_mean_shift_threshold=0.20,
    ).assess()

    analysis = result["feature_analysis"]["feature"]["drift_analysis"]

    assert analysis["relative_mean_shift"] == 0.2
    assert analysis["requires_review"] is False


def test_zero_reference_and_zero_current_mean_has_no_shift():
    """Two zero means should produce zero relative movement."""
    reference = pd.DataFrame(
        {"feature": [-1, 1]}
    )

    current = pd.DataFrame(
        {"feature": [-2, 2]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    analysis = result["feature_analysis"]["feature"]["drift_analysis"]

    assert analysis["reference_mean"] == 0.0
    assert analysis["current_mean"] == 0.0
    assert analysis["relative_mean_shift"] == 0.0


def test_zero_reference_mean_with_nonzero_current_mean_is_flaggable():
    """Movement away from a zero reference mean should be visible."""
    reference = pd.DataFrame(
        {"feature": [-1, 1]}
    )

    current = pd.DataFrame(
        {"feature": [1, 3]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
        numeric_mean_shift_threshold=0.20,
    ).assess()

    analysis = result["feature_analysis"]["feature"]["drift_analysis"]

    assert analysis["reference_mean"] == 0.0
    assert analysis["current_mean"] == 2.0
    assert analysis["relative_mean_shift"] == 1.0
    assert analysis["requires_review"] is True


def test_identical_categorical_distribution_has_no_drift():
    """Identical category proportions should not require review."""
    reference = pd.DataFrame(
        {"segment": ["A", "A", "B", "B"]}
    )

    current = pd.DataFrame(
        {"segment": ["A", "A", "B", "B"]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    analysis = result["feature_analysis"]["segment"]["drift_analysis"]

    assert analysis["maximum_distribution_change"] == 0.0
    assert analysis["requires_review"] is False
    assert result["review_required"] is False


def test_categorical_distribution_shift_requires_review():
    """Large category-proportion movement should be flagged."""
    reference = pd.DataFrame(
        {"segment": ["A"] * 8 + ["B"] * 2}
    )

    current = pd.DataFrame(
        {"segment": ["A"] * 4 + ["B"] * 6}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
        categorical_distribution_threshold=0.20,
    ).assess()

    analysis = result["feature_analysis"]["segment"]["drift_analysis"]

    assert analysis["maximum_distribution_change"] == 0.4
    assert analysis["requires_review"] is True
    assert result["features_requiring_review"] == ["segment"]


def test_new_category_is_included_in_distribution_analysis():
    """Categories appearing only in current data should remain visible."""
    reference = pd.DataFrame(
        {"segment": ["A", "A", "B", "B"]}
    )

    current = pd.DataFrame(
        {"segment": ["A", "B", "C", "C"]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    changes = (
        result["feature_analysis"]["segment"]["drift_analysis"][
            "category_changes"
        ]
    )

    assert "C" in changes
    assert changes["C"]["reference_rate"] == 0.0
    assert changes["C"]["current_rate"] == 0.5


def test_categorical_shift_exactly_on_threshold_is_not_flagged():
    """Distribution movement equal to the threshold should not flag."""
    reference = pd.DataFrame(
        {"segment": ["A"] * 6 + ["B"] * 4}
    )

    current = pd.DataFrame(
        {"segment": ["A"] * 4 + ["B"] * 6}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
        categorical_distribution_threshold=0.20,
    ).assess()

    analysis = result["feature_analysis"]["segment"]["drift_analysis"]

    assert analysis["maximum_distribution_change"] == 0.2
    assert analysis["requires_review"] is False


def test_missing_data_summary_reports_change():
    """Missingness movement should be reported separately."""
    reference = pd.DataFrame(
        {"feature": [1.0, 2.0, None, 4.0]}
    )

    current = pd.DataFrame(
        {"feature": [1.0, None, None, 4.0]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    missing = result["feature_analysis"]["feature"]["missing_data"]

    assert missing["reference_missing_count"] == 1
    assert missing["current_missing_count"] == 2
    assert missing["reference_missing_rate"] == 0.25
    assert missing["current_missing_rate"] == 0.5
    assert missing["missing_rate_change"] == 0.25


def test_incompatible_feature_types_require_review():
    """Incompatible shared feature types should be surfaced."""
    reference = pd.DataFrame(
        {"feature": [1, 2, 3]}
    )

    current = pd.DataFrame(
        {"feature": ["A", "B", "C"]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    feature = result["feature_analysis"]["feature"]

    assert feature["feature_type"] == "incompatible"
    assert feature["drift_analysis"] is None
    assert result["incompatible_features"] == ["feature"]
    assert result["review_required"] is True


def test_reference_only_column_requires_review():
    """A feature missing from current data should require review."""
    reference = pd.DataFrame(
        {
            "shared": [1, 2, 3],
            "old_feature": [4, 5, 6],
        }
    )

    current = pd.DataFrame(
        {
            "shared": [1, 2, 3],
        }
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    assert result["column_alignment"]["reference_only_columns"] == [
        "old_feature"
    ]

    assert result["review_required"] is True


def test_current_only_column_requires_review():
    """A newly appearing feature should require review."""
    reference = pd.DataFrame(
        {
            "shared": [1, 2, 3],
        }
    )

    current = pd.DataFrame(
        {
            "shared": [1, 2, 3],
            "new_feature": [4, 5, 6],
        }
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    assert result["column_alignment"]["current_only_columns"] == [
        "new_feature"
    ]

    assert result["review_required"] is True


def test_multiple_feature_types_are_analysed_together():
    """Numeric and categorical features should coexist in one report."""
    reference = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "segment": ["A", "A", "B", "B"],
        }
    )

    current = pd.DataFrame(
        {
            "age": [30, 40, 50, 60],
            "segment": ["A", "B", "B", "B"],
        }
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    assert result["features_analyzed"] == 2
    assert result["feature_analysis"]["age"]["feature_type"] == "numeric"

    assert (
        result["feature_analysis"]["segment"]["feature_type"]
        == "categorical"
    )


def test_assessment_returns_expected_top_level_structure():
    """Drift assessment should expose a stable output contract."""
    reference = pd.DataFrame(
        {"feature": [1, 2, 3]}
    )

    current = pd.DataFrame(
        {"feature": [1, 2, 3]}
    )

    result = DataDriftAnalyzer(
        reference,
        current,
    ).assess()

    assert set(result.keys()) == {
        "numeric_mean_shift_threshold",
        "categorical_distribution_threshold",
        "column_alignment",
        "features_analyzed",
        "feature_analysis",
        "features_requiring_review",
        "incompatible_features",
        "review_required",
    }
