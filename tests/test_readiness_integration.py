import pandas as pd

from trustlens.readiness.assessor import AIReadinessAssessor


def test_integrated_readiness_assessment_with_target():
    """Integrated assessment should run all readiness components."""
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C005",
                "C006",
                "C007",
                "C008",
                "C009",
                "C010",
            ],
            "age": [25, 31, 29, 44, 37, 52, 41, 33, 28, 46],
            "income": [
                32000,
                41000,
                38000,
                62000,
                54000,
                71000,
                59000,
                47000,
                36000,
                65000,
            ],
            "region": [
                "north",
                "south",
                "north",
                "west",
                "south",
                "west",
                "north",
                "south",
                "west",
                "north",
            ],
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        }
    )

    assessor = AIReadinessAssessor(
        data,
        target_column="target",
    )

    report = assessor.assess()

    assert report["dataset"]["rows"] == 10
    assert report["dataset"]["columns"] == 5
    assert report["target_column"] == "target"

    assert "target" not in report["feature_columns"]
    assert report["feature_count"] == 4

    assert "feature_suitability" in report
    assert "class_imbalance" in report
    assert "leakage_risk" in report
    assert "readiness_summary" in report


def test_feature_suitability_is_integrated():
    """Feature suitability should be exposed through the core assessor."""
    data = pd.DataFrame(
        {
            "constant_feature": [1, 1, 1, 1, 1],
            "useful_feature": [10, 20, 30, 40, 50],
            "target": [0, 0, 0, 1, 1],
        }
    )

    report = AIReadinessAssessor(
        data,
        target_column="target",
    ).assess()

    suitability = report["feature_suitability"]

    assert suitability["feature_count"] == 2
    assert "constant_feature" in suitability["features_requiring_review"]

    constant_result = suitability["features"]["constant_feature"]

    assert constant_result["status"] == "review"
    assert "constant_feature" in constant_result["issues"]


def test_class_imbalance_is_integrated():
    """Class imbalance should be detected through the core assessor."""
    data = pd.DataFrame(
        {
            "feature": list(range(10)),
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    report = AIReadinessAssessor(
        data,
        target_column="target",
    ).assess()

    imbalance = report["class_imbalance"]

    assert imbalance["applicable"] is True
    assert imbalance["is_imbalanced"] is True
    assert 1 in imbalance["minority_classes"]

    assert (
        report["readiness_summary"]["class_imbalance_detected"]
        is True
    )

    assert (
        "class_imbalance"
        in report["readiness_summary"]["review_reasons"]
    )


def test_data_leakage_is_integrated():
    """Leakage warnings should be exposed through the core assessor."""
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40, 50, 60],
            "target_copy": [0, 1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    report = AIReadinessAssessor(
        data,
        target_column="target",
    ).assess()

    leakage = report["leakage_risk"]

    assert leakage["applicable"] is True
    assert "target_copy" in leakage["duplicate_target_features"]
    assert leakage["risk_level"] == "high"

    assert (
        report["readiness_summary"]["leakage_risk_level"]
        == "high"
    )

    assert (
        "data_leakage"
        in report["readiness_summary"]["review_reasons"]
    )


def test_readiness_summary_combines_multiple_risks():
    """Summary should combine issues detected by different analyzers."""
    data = pd.DataFrame(
        {
            "constant_feature": [1] * 10,
            "target_copy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    report = AIReadinessAssessor(
        data,
        target_column="target",
    ).assess()

    summary = report["readiness_summary"]

    assert summary["status"] == "review"

    assert "feature_suitability" in summary["review_reasons"]
    assert "class_imbalance" in summary["review_reasons"]
    assert "data_leakage" in summary["review_reasons"]

    assert summary["features_requiring_review"] >= 1
    assert summary["class_imbalance_detected"] is True
    assert summary["leakage_risk_level"] == "high"


def test_assessment_without_target():
    """Target-dependent analyses should be not applicable without a target."""
    data = pd.DataFrame(
        {
            "age": [25, 31, 29, 44],
            "income": [32000, 41000, 38000, 62000],
            "region": ["north", "south", "north", "west"],
        }
    )

    report = AIReadinessAssessor(data).assess()

    assert report["target_column"] is None
    assert report["feature_count"] == 3

    assert report["class_imbalance"]["applicable"] is False
    assert report["leakage_risk"]["applicable"] is False

    assert (
        report["readiness_summary"]["class_imbalance_detected"]
        is None
    )

    assert (
        report["readiness_summary"]["leakage_risk_level"]
        is None
    )


def test_missing_values_remain_available_in_integrated_report():
    """Foundational missing-value analysis should remain available."""
    data = pd.DataFrame(
        {
            "age": [25, None, 31, 40],
            "income": [50000, 62000, None, 70000],
            "target": [0, 0, 1, 1],
        }
    )

    report = AIReadinessAssessor(
        data,
        target_column="target",
    ).assess()

    missing = report["missing_values"]

    assert missing["total_missing_values"] == 2
    assert missing["missing_by_column"]["age"] == 1
    assert missing["missing_by_column"]["income"] == 1

    assert set(missing["columns_with_missing_values"]) == {
        "age",
        "income",
    }


def test_integrated_report_has_expected_top_level_structure():
    """Integrated report should expose the complete readiness structure."""
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": ["a", "b", "a", "b"],
            "target": [0, 0, 1, 1],
        }
    )

    report = AIReadinessAssessor(
        data,
        target_column="target",
    ).assess()

    expected_sections = {
        "dataset",
        "target_column",
        "feature_count",
        "feature_columns",
        "feature_types",
        "missing_values",
        "feature_suitability",
        "class_imbalance",
        "leakage_risk",
        "readiness_summary",
    }

    assert set(report.keys()) == expected_sections
