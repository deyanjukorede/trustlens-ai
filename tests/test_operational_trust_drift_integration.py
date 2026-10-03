import pandas as pd

from trustlens.operational_trust.assessor import OperationalTrustAssessor


def test_drift_analysis_is_unavailable_without_reference_data():
    """Drift analysis should not run without a reference dataset."""
    data = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "segment": ["A", "A", "B", "B"],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    assert report["data_drift"] is None

    assert (
        report["analysis_availability"]["data_drift"]["available"]
        is False
    )

    assert report["operational_trust_summary"] == {
        "status": "no_review_indicators",
        "review_required": False,
        "review_reasons": [],
    }


def test_reference_data_triggers_drift_analysis():
    """Supplying reference data should activate drift analysis."""
    reference_data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    assert report["data_drift"] is not None

    assert (
        report["analysis_availability"]["data_drift"]["available"]
        is True
    )

    assert report["data_drift"]["features_analyzed"] == 1


def test_no_drift_indicator_produces_no_review_reason():
    """Stable comparison data should not create a drift review reason."""
    reference_data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "segment": ["A", "A", "B", "B"],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "segment": ["A", "A", "B", "B"],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    assert report["data_drift"]["review_required"] is False

    assert report["operational_trust_summary"] == {
        "status": "no_review_indicators",
        "review_required": False,
        "review_reasons": [],
    }


def test_numeric_drift_propagates_to_operational_summary():
    """Numeric drift should propagate into the integrated summary."""
    reference_data = pd.DataFrame(
        {
            "feature": [10, 10, 10, 10],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [15, 15, 15, 15],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    drift = report["data_drift"]

    assert drift["features_requiring_review"] == ["feature"]
    assert drift["review_required"] is True

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": ["data_drift"],
    }


def test_categorical_drift_propagates_to_operational_summary():
    """Categorical distribution movement should propagate to summary."""
    reference_data = pd.DataFrame(
        {
            "segment": ["A"] * 8 + ["B"] * 2,
        }
    )

    current_data = pd.DataFrame(
        {
            "segment": ["A"] * 4 + ["B"] * 6,
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    drift = report["data_drift"]

    assert drift["features_requiring_review"] == ["segment"]
    assert drift["review_required"] is True

    assert (
        report["operational_trust_summary"]["review_reasons"]
        == ["data_drift"]
    )


def test_schema_change_propagates_to_operational_summary():
    """Schema differences should produce an integrated review reason."""
    reference_data = pd.DataFrame(
        {
            "shared": [1, 2, 3],
            "old_feature": [4, 5, 6],
        }
    )

    current_data = pd.DataFrame(
        {
            "shared": [1, 2, 3],
            "new_feature": [7, 8, 9],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    drift = report["data_drift"]

    assert drift["column_alignment"]["reference_only_columns"] == [
        "old_feature"
    ]

    assert drift["column_alignment"]["current_only_columns"] == [
        "new_feature"
    ]

    assert drift["review_required"] is True

    assert report["operational_trust_summary"]["status"] == "review"
    assert (
        report["operational_trust_summary"]["review_required"]
        is True
    )
    assert (
        "data_drift"
        in report["operational_trust_summary"]["review_reasons"]
    )


def test_incompatible_feature_type_propagates_to_summary():
    """Feature type incompatibility should remain visible centrally."""
    reference_data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": ["A", "B", "C"],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    drift = report["data_drift"]

    assert drift["incompatible_features"] == ["feature"]
    assert drift["review_required"] is True

    assert (
        report["operational_trust_summary"]["review_reasons"]
        == ["data_drift"]
    )


def test_custom_numeric_threshold_is_used_by_integrated_drift():
    """Assessor should pass custom numeric thresholds to drift analysis."""
    reference_data = pd.DataFrame(
        {
            "feature": [10, 10, 10, 10],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [13, 13, 13, 13],
        }
    )

    default_report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    relaxed_report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
        numeric_mean_shift_threshold=0.40,
    ).assess()

    assert (
        default_report["data_drift"]["feature_analysis"]["feature"][
            "drift_analysis"
        ]["relative_mean_shift"]
        == 0.3
    )

    assert default_report["data_drift"]["review_required"] is True
    assert relaxed_report["data_drift"]["review_required"] is False


def test_custom_categorical_threshold_is_used_by_integrated_drift():
    """Assessor should pass custom categorical thresholds to drift."""
    reference_data = pd.DataFrame(
        {
            "segment": ["A"] * 7 + ["B"] * 3,
        }
    )

    current_data = pd.DataFrame(
        {
            "segment": ["A"] * 4 + ["B"] * 6,
        }
    )

    default_report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    relaxed_report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
        categorical_distribution_threshold=0.40,
    ).assess()

    assert default_report["data_drift"]["review_required"] is True
    assert relaxed_report["data_drift"]["review_required"] is False


def test_drift_output_is_preserved_inside_integrated_report():
    """Integrated assessor should retain detailed drift evidence."""
    reference_data = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "segment": ["A", "A", "B", "B"],
        }
    )

    current_data = pd.DataFrame(
        {
            "age": [30, 40, 50, 60],
            "segment": ["A", "B", "B", "B"],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    drift = report["data_drift"]

    assert "column_alignment" in drift
    assert "feature_analysis" in drift
    assert "features_requiring_review" in drift
    assert "incompatible_features" in drift
    assert "review_required" in drift

    assert drift["features_analyzed"] == 2
    assert drift["feature_analysis"]["age"]["feature_type"] == "numeric"

    assert (
        drift["feature_analysis"]["segment"]["feature_type"]
        == "categorical"
    )
