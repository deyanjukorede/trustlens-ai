import pandas as pd

from trustlens.operational_trust.assessor import OperationalTrustAssessor


def test_drift_analysis_is_unavailable_without_reference_data():
    """Drift analysis should not run without reference data."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    assert (
        report["analysis_availability"]["data_drift"]["available"]
        is False
    )

    assert report["data_drift"] is None


def test_reference_data_triggers_drift_analysis():
    """Reference data should enable integrated drift analysis."""
    reference_data = pd.DataFrame(
        {
            "feature": [10, 11, 12, 13],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [10, 11, 12, 13],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    assert (
        report["analysis_availability"]["data_drift"]["available"]
        is True
    )

    assert report["data_drift"] is not None
    assert report["data_drift"]["features_analyzed"] == 1


def test_no_drift_indicator_produces_no_review_reason():
    """Stable comparison data should not add a drift review reason."""
    reference_data = pd.DataFrame(
        {
            "feature": [10, 11, 12, 13],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [10, 11, 12, 13],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    assert report["data_drift"]["review_required"] is False

    assert (
        "data_drift"
        not in report["operational_trust_summary"]["review_reasons"]
    )


def test_numeric_drift_propagates_to_operational_summary():
    """Numeric drift should propagate into the integrated summary."""
    reference_data = pd.DataFrame(
        {
            "feature": [10, 11, 12, 13],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [20, 21, 22, 23],
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
            "segment": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
        }
    )

    current_data = pd.DataFrame(
        {
            "segment": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "B",
                "C",
                "C",
            ],
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
        report["data_stability"]["review_required"]
        is False
    )

    assert (
        report["operational_trust_summary"]["review_reasons"]
        == ["data_drift"]
    )


def test_schema_change_propagates_to_operational_summary():
    """Schema changes should propagate into the integrated summary."""
    reference_data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "old_feature": [10, 20, 30],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "new_feature": [10, 20, 30],
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

    assert (
        "data_drift"
        in report["operational_trust_summary"]["review_reasons"]
    )


def test_incompatible_feature_type_propagates_to_summary():
    """Feature type incompatibility should remain visible centrally."""
    reference_data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": ["A", "B", "A", "B"],
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
        report["data_stability"]["review_required"]
        is False
    )

    assert (
        report["operational_trust_summary"]["review_reasons"]
        == ["data_drift"]
    )


def test_custom_numeric_threshold_is_used_by_integrated_drift():
    """Custom numeric drift threshold should flow through assessor."""
    reference_data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature": [11, 21, 31, 41],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
        numeric_mean_shift_threshold=0.50,
    ).assess()

    assert (
        report["data_drift"]["thresholds"][
            "numeric_mean_shift_threshold"
        ]
        == 0.50
    )

    assert report["data_drift"]["review_required"] is False


def test_custom_categorical_threshold_is_used_by_integrated_drift():
    """Custom categorical drift threshold should flow through assessor."""
    reference_data = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B"],
        }
    )

    current_data = pd.DataFrame(
        {
            "segment": ["A", "A", "A", "B"],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
        categorical_distribution_threshold=0.50,
    ).assess()

    assert (
        report["data_drift"]["thresholds"][
            "categorical_distribution_threshold"
        ]
        == 0.50
    )

    assert report["data_drift"]["review_required"] is False


def test_drift_output_is_preserved_inside_integrated_report():
    """Integrated assessment should preserve detailed drift evidence."""
    reference_data = pd.DataFrame(
        {
            "numeric_feature": [10, 11, 12, 13],
            "category": ["A", "B", "A", "B"],
        }
    )

    current_data = pd.DataFrame(
        {
            "numeric_feature": [20, 21, 22, 23],
            "category": ["A", "A", "A", "B"],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    drift = report["data_drift"]

    assert "thresholds" in drift
    assert "column_alignment" in drift
    assert "feature_analysis" in drift
    assert "features_requiring_review" in drift
    assert "incompatible_features" in drift
    assert "review_required" in drift

    assert "numeric_feature" in drift["feature_analysis"]
    assert "category" in drift["feature_analysis"]
