import pandas as pd

from trustlens.operational_trust.assessor import OperationalTrustAssessor


def complete_reproducibility_metadata():
    """Provide complete reproducibility evidence."""
    return {
        "dataset_version": "v1.0",
        "code_version": "commit-abc123",
        "random_seed": 42,
        "environment": "python-3.12",
        "data_source": "validated-source",
        "lineage": "source-to-model-pipeline",
        "execution_id": "run-001",
    }


def complete_monitoring_metadata():
    """Provide complete monitoring evidence."""
    return {
        "monitoring_metrics": ["data_drift", "missingness"],
        "alerting_rules": {"data_drift": "review_threshold"},
        "monitoring_owner": "ml-platform",
        "review_cadence": "monthly",
        "logging_enabled": True,
        "incident_process": "operational-runbook",
        "reassessment_triggers": ["material_data_change"],
        "monitoring_history": ["2026-10-review"],
    }


def stable_data():
    """Return a dataset without expected stability indicators."""
    return pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "score": [10, 20, 30, 40],
        }
    )


def assess_with_complete_evidence(data, reference_data=None):
    """Run assessment with complete optional evidence."""
    return OperationalTrustAssessor(
        data,
        reference_data=reference_data,
        reproducibility_metadata=complete_reproducibility_metadata(),
        identifier_column="record_id",
        monitoring_metadata=complete_monitoring_metadata(),
    ).assess()


def test_overview_is_present_in_integrated_report():
    """The integrated report should expose its dimension overview."""
    report = OperationalTrustAssessor(stable_data()).assess()

    assert "operational_trust_overview" in report
    assert isinstance(report["operational_trust_overview"], dict)


def test_overview_contains_all_four_dimensions():
    """All Operational Trust dimensions should be represented."""
    report = OperationalTrustAssessor(stable_data()).assess()

    dimensions = report["operational_trust_overview"]["dimensions"]

    assert list(dimensions) == [
        "data_drift",
        "data_stability",
        "reproducibility",
        "monitoring_readiness",
    ]


def test_drift_is_unavailable_without_reference_data():
    """Missing reference data should make drift unavailable."""
    report = OperationalTrustAssessor(stable_data()).assess()
    overview = report["operational_trust_overview"]

    assert overview["total_dimensions"] == 4
    assert overview["available_dimension_count"] == 3
    assert overview["available_dimensions"] == [
        "data_stability",
        "reproducibility",
        "monitoring_readiness",
    ]
    assert overview["unavailable_dimensions"] == ["data_drift"]

    assert overview["dimensions"]["data_drift"] == {
        "available": False,
        "review_required": None,
    }


def test_all_dimensions_available_with_reference_data():
    """Reference data should enable all four dimensions."""
    data = stable_data()

    report = OperationalTrustAssessor(
        data,
        reference_data=data.copy(),
    ).assess()

    overview = report["operational_trust_overview"]

    assert overview["total_dimensions"] == 4
    assert overview["available_dimension_count"] == 4
    assert overview["unavailable_dimensions"] == []

    assert all(
        result["available"]
        for result in overview["dimensions"].values()
    )


def test_missing_optional_evidence_requires_review():
    """Missing optional evidence should remain visible centrally."""
    report = OperationalTrustAssessor(stable_data()).assess()
    overview = report["operational_trust_overview"]

    assert overview["dimensions_requiring_review"] == [
        "reproducibility",
        "monitoring_readiness",
    ]
    assert overview["review_dimension_count"] == 2
    assert overview["review_required"] is True


def test_complete_evidence_can_clear_review_indicators():
    """Complete evidence and stable data can clear review indicators."""
    report = assess_with_complete_evidence(stable_data())
    overview = report["operational_trust_overview"]

    assert overview["dimensions_requiring_review"] == []
    assert overview["review_dimension_count"] == 0
    assert overview["review_required"] is False

    assert report["operational_trust_summary"] == {
        "status": "no_review_indicators",
        "review_required": False,
        "review_reasons": [],
    }


def test_complete_evidence_with_reference_data():
    """All four dimensions can be assessed without review indicators."""
    data = stable_data()
    report = assess_with_complete_evidence(
        data,
        reference_data=data.copy(),
    )

    overview = report["operational_trust_overview"]

    assert overview["available_dimension_count"] == 4
    assert overview["review_dimension_count"] == 0
    assert overview["review_required"] is False


def test_numeric_drift_is_visible_in_overview():
    """Numeric drift should propagate into the dimension overview."""
    reference_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "score": [10, 20, 30, 40],
        }
    )
    current_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "score": [30, 60, 90, 120],
        }
    )

    report = assess_with_complete_evidence(
        current_data,
        reference_data=reference_data,
    )
    overview = report["operational_trust_overview"]

    assert overview["dimensions_requiring_review"] == ["data_drift"]
    assert overview["review_dimension_count"] == 1
    assert overview["review_required"] is True


def test_stability_review_is_visible_in_overview():
    """Stability findings should propagate independently."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "score": [10, 10, 10, 10],
        }
    )

    report = assess_with_complete_evidence(data)
    overview = report["operational_trust_overview"]

    assert overview["dimensions_requiring_review"] == [
        "data_stability"
    ]
    assert overview["review_dimension_count"] == 1


def test_multiple_review_dimensions_can_coexist():
    """Different review indicators should remain independently visible."""
    reference_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "score": [10, 20, 30, 40],
        }
    )
    current_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "score": [100, 100, 100, 100],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    overview = report["operational_trust_overview"]

    assert overview["dimensions_requiring_review"] == [
        "data_drift",
        "data_stability",
        "reproducibility",
        "monitoring_readiness",
    ]
    assert overview["review_dimension_count"] == 4


def test_overview_agrees_with_original_summary():
    """Overview and legacy summary should report consistent findings."""
    report = OperationalTrustAssessor(stable_data()).assess()

    overview = report["operational_trust_overview"]
    summary = report["operational_trust_summary"]

    assert overview["review_required"] == summary["review_required"]
    assert (
        overview["dimensions_requiring_review"]
        == summary["review_reasons"]
    )


def test_overview_interpretation_reflects_review():
    """Review indicators should produce a human-review interpretation."""
    report = OperationalTrustAssessor(stable_data()).assess()
    overview = report["operational_trust_overview"]

    assert "human review" in overview["interpretation"].lower()


def test_no_review_interpretation_is_evidence_limited():
    """Absence of indicators should not imply proven trustworthiness."""
    report = assess_with_complete_evidence(stable_data())
    overview = report["operational_trust_overview"]

    assert "no review indicators" in (
        overview["interpretation"].lower()
    )


def test_scope_note_avoids_unsupported_trust_claims():
    """The overview should explicitly limit its conclusions."""
    report = OperationalTrustAssessor(stable_data()).assess()
    overview = report["operational_trust_overview"]

    assert "does not independently establish" in (
        overview["scope_note"].lower()
    )
    assert "trustworthy" in overview["scope_note"].lower()


def test_unavailable_drift_is_not_counted_as_review():
    """Unavailable analysis must not be treated as a drift finding."""
    report = OperationalTrustAssessor(stable_data()).assess()
    overview = report["operational_trust_overview"]

    assert "data_drift" in overview["unavailable_dimensions"]
    assert "data_drift" not in overview["dimensions_requiring_review"]
    assert overview["dimensions"]["data_drift"]["review_required"] is None
