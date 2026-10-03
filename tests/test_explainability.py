import pandas as pd
import pytest

from trustlens.responsible_ai.explainability import ExplainabilityAnalyzer


def test_analyzer_initialises_with_valid_dataframe():
    """Analyzer should initialise with a valid non-empty DataFrame."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "income": [30000, 40000, 50000],
        }
    )

    analyzer = ExplainabilityAnalyzer(data)

    assert analyzer.data is data
    assert analyzer.high_cardinality_threshold == 20
    assert analyzer.missingness_threshold == 0.50


def test_analyzer_rejects_non_dataframe():
    """Analyzer should reject inputs that are not pandas DataFrames."""
    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
        ExplainabilityAnalyzer(
            {
                "age": [25, 35],
            }
        )


def test_analyzer_rejects_empty_dataframe():
    """Analyzer should reject an empty DataFrame."""
    data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        ExplainabilityAnalyzer(data)


def test_analyzer_validates_target_column():
    """Configured target column must exist."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="target_column",
    ):
        ExplainabilityAnalyzer(
            data,
            target_column="target",
        )


def test_analyzer_validates_prediction_column():
    """Configured prediction column must exist."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="prediction_column",
    ):
        ExplainabilityAnalyzer(
            data,
            prediction_column="prediction",
        )


def test_analyzer_rejects_non_list_sensitive_attributes():
    """Sensitive attributes should be supplied as a list."""
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "feature": [1, 2],
        }
    )

    with pytest.raises(
        TypeError,
        match="sensitive_attributes must be a list",
    ):
        ExplainabilityAnalyzer(
            data,
            sensitive_attributes="group",
        )


def test_analyzer_validates_sensitive_attributes():
    """Configured sensitive attributes must exist."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="sensitive attributes",
    ):
        ExplainabilityAnalyzer(
            data,
            sensitive_attributes=["group"],
        )


def test_analyzer_validates_high_cardinality_threshold():
    """High-cardinality threshold must be a positive integer."""
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
            ExplainabilityAnalyzer(
                data,
                high_cardinality_threshold=value,
            )


def test_analyzer_validates_missingness_threshold():
    """Missingness threshold should be between zero and one."""
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
            ExplainabilityAnalyzer(
                data,
                missingness_threshold=value,
            )


def test_missingness_threshold_boundaries_are_allowed():
    """Zero and one should be valid missingness thresholds."""
    data = pd.DataFrame(
        {
            "feature": [1, 2],
        }
    )

    zero = ExplainabilityAnalyzer(
        data,
        missingness_threshold=0,
    )

    one = ExplainabilityAnalyzer(
        data,
        missingness_threshold=1,
    )

    assert zero.missingness_threshold == 0.0
    assert one.missingness_threshold == 1.0


def test_excluded_columns_include_configured_context_columns():
    """Target, prediction and sensitive attributes should be excluded."""
    data = pd.DataFrame(
        {
            "age": [25, 35],
            "income": [30000, 40000],
            "group": ["A", "B"],
            "target": [1, 0],
            "prediction": [1, 1],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    assert analyzer.excluded_columns() == [
        "target",
        "prediction",
        "group",
    ]


def test_candidate_features_exclude_context_columns():
    """Only non-context columns should remain as candidates."""
    data = pd.DataFrame(
        {
            "age": [25, 35],
            "income": [30000, 40000],
            "group": ["A", "B"],
            "target": [1, 0],
            "prediction": [1, 1],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    assert analyzer.candidate_features() == [
        "age",
        "income",
    ]


def test_numeric_feature_is_classified_correctly():
    """Numeric features should be identified."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45],
        }
    )

    analyzer = ExplainabilityAnalyzer(data)

    result = analyzer.feature_analysis()

    assert result["age"]["feature_type"] == "numeric"


def test_categorical_feature_is_classified_correctly():
    """Object/string features should be categorical."""
    data = pd.DataFrame(
        {
            "city": ["London", "Sheffield", "Leeds"],
        }
    )

    analyzer = ExplainabilityAnalyzer(data)

    result = analyzer.feature_analysis()

    assert result["city"]["feature_type"] == "categorical"


def test_boolean_feature_is_classified_correctly():
    """Boolean features should be identified before numeric types."""
    data = pd.DataFrame(
        {
            "active": [True, False, True],
        }
    )

    analyzer = ExplainabilityAnalyzer(data)

    result = analyzer.feature_analysis()

    assert result["active"]["feature_type"] == "boolean"


def test_datetime_feature_is_classified_correctly():
    """Datetime features should be identified."""
    data = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
        }
    )

    analyzer = ExplainabilityAnalyzer(data)

    result = analyzer.feature_analysis()

    assert result["event_date"]["feature_type"] == "datetime"


def test_feature_type_summary_counts_feature_types():
    """Feature-type summary should count candidate feature types."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "city": ["A", "B", "C"],
            "active": [True, False, True],
            "event_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
        }
    )

    analyzer = ExplainabilityAnalyzer(data)

    feature_results = analyzer.feature_analysis()

    summary = analyzer.feature_type_summary(
        feature_results
    )

    assert summary["numeric"] == 1
    assert summary["categorical"] == 1
    assert summary["boolean"] == 1
    assert summary["datetime"] == 1
    assert summary["other"] == 0


def test_constant_feature_is_detected():
    """Features with one observed value should be flagged as constant."""
    data = pd.DataFrame(
        {
            "constant": [5, 5, 5, 5],
            "variable": [1, 2, 3, 4],
        }
    )

    analyzer = ExplainabilityAnalyzer(data)

    result = analyzer.feature_analysis()

    assert result["constant"]["constant_feature"] is True

    assert (
        "constant_feature"
        in result["constant"]["indicators"]
    )

    assert result["constant"]["requires_review"] is True

    assert result["variable"]["constant_feature"] is False


def test_all_missing_feature_is_treated_as_constant_and_missing():
    """An all-missing feature should expose both structural issues."""
    data = pd.DataFrame(
        {
            "feature": [None, None, None, None],
        }
    )

    analyzer = ExplainabilityAnalyzer(data)

    result = analyzer.feature_analysis()["feature"]

    assert result["unique_values"] == 0
    assert result["constant_feature"] is True
    assert result["high_missingness"] is True
    assert result["missing_rate"] == 1.0

    assert "constant_feature" in result["indicators"]
    assert "high_missingness" in result["indicators"]


def test_high_missingness_is_detected():
    """Features at or above the threshold should be flagged."""
    data = pd.DataFrame(
        {
            "feature": [1, None, None, 4],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        missingness_threshold=0.50,
    )

    result = analyzer.feature_analysis()["feature"]

    assert result["missing_count"] == 2
    assert result["missing_rate"] == 0.5
    assert result["high_missingness"] is True

    assert "high_missingness" in result["indicators"]


def test_missingness_below_threshold_is_not_flagged():
    """Missingness below the configured threshold should not flag."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, None, 4],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        missingness_threshold=0.50,
    )

    result = analyzer.feature_analysis()["feature"]

    assert result["missing_rate"] == 0.25
    assert result["high_missingness"] is False


def test_zero_missingness_threshold_flags_complete_feature():
    """A zero threshold should flag even zero missingness."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        missingness_threshold=0,
    )

    result = analyzer.feature_analysis()["feature"]

    assert result["missing_rate"] == 0.0
    assert result["high_missingness"] is True


def test_high_cardinality_categorical_feature_is_detected():
    """Categorical cardinality above the threshold should be flagged."""
    data = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "C",
                "D",
            ],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        high_cardinality_threshold=3,
    )

    result = analyzer.feature_analysis()["category"]

    assert result["unique_values"] == 4
    assert result["high_cardinality"] is True

    assert "high_cardinality" in result["indicators"]


def test_exact_high_cardinality_threshold_is_not_flagged():
    """Cardinality equal to the threshold should not be flagged."""
    data = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "C",
            ],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        high_cardinality_threshold=3,
    )

    result = analyzer.feature_analysis()["category"]

    assert result["unique_values"] == 3
    assert result["high_cardinality"] is False


def test_numeric_feature_is_not_flagged_for_high_cardinality():
    """High-cardinality indicator should apply to categorical data."""
    data = pd.DataFrame(
        {
            "numeric_id": [1, 2, 3, 4, 5],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        high_cardinality_threshold=2,
    )

    result = analyzer.feature_analysis()["numeric_id"]

    assert result["feature_type"] == "numeric"
    assert result["high_cardinality"] is False


def test_review_summary_collects_feature_indicators():
    """Review summary should identify all flagged features."""
    data = pd.DataFrame(
        {
            "constant": [1, 1, 1, 1],
            "missing": [1, None, None, 4],
            "category": ["A", "B", "C", "D"],
            "clean": [1, 2, 3, 4],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        high_cardinality_threshold=3,
        missingness_threshold=0.50,
    )

    feature_results = analyzer.feature_analysis()

    summary = analyzer.review_summary(
        feature_results
    )

    assert summary["constant_features"] == [
        "constant"
    ]

    assert summary["high_missingness_features"] == [
        "missing"
    ]

    assert summary["high_cardinality_features"] == [
        "category"
    ]

    assert set(
        summary["features_requiring_review"]
    ) == {
        "constant",
        "missing",
        "category",
    }

    assert summary["review_required"] is True


def test_clean_features_produce_ready_status():
    """Usable features without indicators should be explanation-ready."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "income": [30000, 40000, 50000, 60000],
        }
    )

    result = ExplainabilityAnalyzer(data).assess()

    readiness = result["explanation_readiness"]

    assert (
        readiness["status"]
        == "ready_for_explanation_analysis"
    )

    assert readiness["candidate_feature_count"] == 2
    assert readiness["usable_feature_count"] == 2

    assert readiness["usable_features"] == [
        "age",
        "income",
    ]

    assert readiness["review_required"] is False
    assert result["review_required"] is False


def test_review_indicator_produces_review_status():
    """A structural indicator should produce review status."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "constant": [1, 1, 1, 1],
        }
    )

    result = ExplainabilityAnalyzer(data).assess()

    readiness = result["explanation_readiness"]

    assert readiness["status"] == "review"
    assert readiness["usable_features"] == ["age"]
    assert readiness["review_required"] is True
    assert result["review_required"] is True


def test_only_unusable_features_produce_limited_readiness():
    """No usable candidate features should produce limited readiness."""
    data = pd.DataFrame(
        {
            "constant": [1, 1, 1, 1],
            "missing": [None, None, None, None],
        }
    )

    result = ExplainabilityAnalyzer(data).assess()

    readiness = result["explanation_readiness"]

    assert (
        readiness["status"]
        == "limited_explanation_readiness"
    )

    assert readiness["candidate_feature_count"] == 2
    assert readiness["usable_feature_count"] == 0
    assert readiness["usable_features"] == []
    assert readiness["review_required"] is True


def test_no_candidate_features_produce_insufficient_status():
    """Context-only datasets should report insufficient features."""
    data = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B"],
            "target": [1, 0, 1, 0],
            "prediction": [1, 0, 0, 0],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    )

    result = analyzer.assess()

    assert result["candidate_features"] == []
    assert result["candidate_feature_count"] == 0
    assert result["feature_analysis"] == {}

    readiness = result["explanation_readiness"]

    assert readiness["status"] == "insufficient_features"
    assert readiness["candidate_feature_count"] == 0
    assert readiness["usable_feature_count"] == 0
    assert readiness["usable_features"] == []
    assert readiness["review_required"] is True

    assert result["review_required"] is True


def test_sensitive_attributes_are_not_analysed_as_candidate_features():
    """Sensitive attributes should remain outside feature analysis."""
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "age": [25, 35, 45, 55],
            "income": [30000, 40000, 50000, 60000],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        sensitive_attributes=["group"],
    )

    result = analyzer.assess()

    assert "group" in result["excluded_columns"]
    assert "group" not in result["candidate_features"]
    assert "group" not in result["feature_analysis"]

    assert result["candidate_features"] == [
        "age",
        "income",
    ]


def test_context_columns_do_not_affect_candidate_feature_count():
    """Excluded context columns should not inflate feature counts."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "income": [30000, 40000, 50000, 60000],
            "group": ["A", "A", "B", "B"],
            "target": [1, 0, 1, 0],
            "prediction": [1, 0, 0, 0],
        }
    )

    result = ExplainabilityAnalyzer(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
    ).assess()

    assert result["candidate_feature_count"] == 2

    assert result["excluded_columns"] == [
        "target",
        "prediction",
        "group",
    ]

    assert result["candidate_features"] == [
        "age",
        "income",
    ]


def test_assess_returns_expected_structure():
    """Complete assessment should return the expected structure."""
    data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "city": [
                "Sheffield",
                "Leeds",
                "Manchester",
                "London",
            ],
            "group": ["A", "A", "B", "B"],
            "target": [1, 0, 1, 0],
            "prediction": [1, 0, 0, 0],
        }
    )

    analyzer = ExplainabilityAnalyzer(
        data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        high_cardinality_threshold=10,
        missingness_threshold=0.50,
    )

    result = analyzer.assess()

    expected_keys = {
        "target_column",
        "prediction_column",
        "sensitive_attributes",
        "high_cardinality_threshold",
        "missingness_threshold",
        "excluded_columns",
        "candidate_features",
        "candidate_feature_count",
        "feature_types",
        "feature_analysis",
        "review_summary",
        "explanation_readiness",
        "review_required",
    }

    assert set(result.keys()) == expected_keys

    assert result["target_column"] == "target"
    assert result["prediction_column"] == "prediction"
    assert result["sensitive_attributes"] == ["group"]

    assert result["high_cardinality_threshold"] == 10
    assert result["missingness_threshold"] == 0.50

    assert result["excluded_columns"] == [
        "target",
        "prediction",
        "group",
    ]

    assert result["candidate_features"] == [
        "age",
        "city",
    ]

    assert result["candidate_feature_count"] == 2

    assert result["feature_types"]["numeric"] == 1
    assert result["feature_types"]["categorical"] == 1

    assert "age" in result["feature_analysis"]
    assert "city" in result["feature_analysis"]

    assert "review_required" in result["review_summary"]
    assert "status" in result["explanation_readiness"]
