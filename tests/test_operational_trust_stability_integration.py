import pandas as pd

from trustlens.operational_trust.assessor import OperationalTrustAssessor


def test_stability_runs_without_reference_data():
    """Stability analysis should run without a reference dataset."""
    data = pd.DataFrame(
        {
            "score": [10, 20, 30, 40],
            "segment": ["A", "B", "A", "B"],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    assert report["data_drift"] is None
    assert report["data_stability"] is not None

    assert (
        report["analysis_availability"]["data_stability"]["available"]
        is True
    )

    assert report["data_stability"]["review_required"] is False

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": ["reproducibility"],
    }


def test_high_missingness_propagates_to_operational_trust_summary():
    """High missingness should propagate through the central assessor."""
    data = pd.DataFrame(
        {
            "feature": [1.0, None, None, None],
            "other": [10, 20, 30, 40],
        }
    )

    report = OperationalTrustAssessor(
        data,
        high_missingness_threshold=0.50,
    ).assess()

    stability = report["data_stability"]

    assert stability["high_missingness_features"] == ["feature"]
    assert stability["review_required"] is True

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": [
            "data_stability",
            "reproducibility",
        ],
    }


def test_constant_feature_propagates_to_operational_trust_summary():
    """Constant features should propagate through the central assessor."""
    data = pd.DataFrame(
        {
            "constant": [7, 7, 7, 7],
            "variable": [1, 2, 3, 4],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    stability = report["data_stability"]

    assert stability["constant_features"] == ["constant"]
    assert "constant_features" in stability["review_reasons"]

    assert report["operational_trust_summary"]["review_required"] is True
    assert report["operational_trust_summary"]["review_reasons"] == [
        "data_stability",
        "reproducibility",
    ]


def test_near_constant_feature_propagates_to_central_assessor():
    """Near-constant features should remain visible after integration."""
    data = pd.DataFrame(
        {
            "feature": ["A"] * 19 + ["B"],
            "measurement": list(range(20)),
        }
    )

    report = OperationalTrustAssessor(
        data,
        near_constant_threshold=0.90,
    ).assess()

    stability = report["data_stability"]

    assert stability["near_constant_features"] == ["feature"]
    assert "near_constant_features" in stability["review_reasons"]

    assert report["operational_trust_summary"]["review_required"] is True
    assert "data_stability" in (
        report["operational_trust_summary"]["review_reasons"]
    )


def test_high_cardinality_propagates_to_central_assessor():
    """Categorical high cardinality should propagate through integration."""
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
            ],
            "score": list(range(10)),
        }
    )

    report = OperationalTrustAssessor(
        data,
        high_cardinality_threshold=0.80,
    ).assess()

    stability = report["data_stability"]

    assert stability["high_cardinality_features"] == [
        "customer_code"
    ]

    assert "high_cardinality" in stability["review_reasons"]
    assert report["operational_trust_summary"]["review_required"] is True


def test_duplicate_pressure_propagates_to_central_assessor():
    """Duplicate pressure should propagate through integration."""
    data = pd.DataFrame(
        {
            "feature": [1, 1, 1, 2, 3],
            "segment": ["A", "A", "A", "B", "C"],
        }
    )

    report = OperationalTrustAssessor(
        data,
        duplicate_rate_threshold=0.20,
    ).assess()

    stability = report["data_stability"]

    assert stability["duplicate_analysis"]["duplicate_rows"] == 2
    assert stability["duplicate_analysis"]["duplicate_rate"] == 0.4
    assert stability["duplicate_analysis"]["requires_review"] is True

    assert "duplicate_pressure" in stability["review_reasons"]
    assert "data_stability" in (
        report["operational_trust_summary"]["review_reasons"]
    )


def test_custom_missingness_threshold_is_passed_to_stability_analyzer():
    """Custom missingness thresholds should reach the analyzer."""
    data = pd.DataFrame(
        {
            "feature": [1.0, 2.0, None, None],
            "other": [1, 2, 3, 4],
        }
    )

    report = OperationalTrustAssessor(
        data,
        high_missingness_threshold=0.40,
    ).assess()

    stability = report["data_stability"]

    assert stability["thresholds"]["high_missingness"] == 0.40
    assert stability["high_missingness_features"] == ["feature"]


def test_custom_near_constant_threshold_is_passed_to_analyzer():
    """Custom concentration thresholds should reach the analyzer."""
    data = pd.DataFrame(
        {
            "feature": [
                "A",
                "A",
                "A",
                "A",
                "B",
            ]
        }
    )

    report = OperationalTrustAssessor(
        data,
        near_constant_threshold=0.70,
        high_cardinality_threshold=1.0,
    ).assess()

    stability = report["data_stability"]

    assert stability["thresholds"]["near_constant"] == 0.70
    assert stability["near_constant_features"] == ["feature"]


def test_custom_cardinality_threshold_is_passed_to_analyzer():
    """Custom cardinality thresholds should reach the analyzer."""
    data = pd.DataFrame(
        {
            "category": ["A", "B", "C", "D", "D"],
        }
    )

    report = OperationalTrustAssessor(
        data,
        high_cardinality_threshold=0.70,
    ).assess()

    stability = report["data_stability"]

    assert stability["thresholds"]["high_cardinality"] == 0.70
    assert stability["high_cardinality_features"] == ["category"]


def test_custom_duplicate_threshold_is_passed_to_analyzer():
    """Custom duplicate thresholds should reach the analyzer."""
    data = pd.DataFrame(
        {
            "feature": [1, 1, 2, 3],
        }
    )

    report = OperationalTrustAssessor(
        data,
        duplicate_rate_threshold=0.20,
    ).assess()

    stability = report["data_stability"]

    assert stability["thresholds"]["duplicate_rate"] == 0.20
    assert stability["duplicate_analysis"]["requires_review"] is True


def test_drift_and_stability_review_reasons_can_coexist():
    """Drift and stability findings should remain independently visible."""
    reference_data = pd.DataFrame(
        {
            "score": [10, 11, 12, 13],
            "segment": ["A", "B", "A", "B"],
        }
    )

    current_data = pd.DataFrame(
        {
            "score": [100, 100, 100, 100],
            "segment": ["A", "A", "A", "A"],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    assert report["data_drift"] is not None
    assert report["data_drift"]["review_required"] is True

    assert report["data_stability"]["review_required"] is True
    assert "constant_features" in (
        report["data_stability"]["review_reasons"]
    )

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": [
            "data_drift",
            "data_stability",
            "reproducibility",
        ],
    }


def test_stability_detail_is_preserved_in_integrated_report():
    """Central integration should preserve detailed stability evidence."""
    data = pd.DataFrame(
        {
            "feature": [1.0, None, None, None],
            "segment": ["A", "B", "A", "B"],
        }
    )

    report = OperationalTrustAssessor(
        data,
        high_missingness_threshold=0.50,
    ).assess()

    stability = report["data_stability"]

    assert "feature_analysis" in stability
    assert "feature_type_summary" in stability
    assert "duplicate_analysis" in stability

    assert (
        stability["feature_analysis"]["feature"]["missingness"][
            "missing_rate"
        ]
        == 0.75
    )


def test_stability_analysis_availability_requires_only_current_data():
    """Stability availability should not depend on reference data."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    availability = OperationalTrustAssessor(
        data
    ).analysis_availability()

    assert availability["data_stability"] == {
        "available": True,
        "requires": ["current_data"],
    }
