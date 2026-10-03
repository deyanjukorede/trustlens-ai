import pandas as pd

from trustlens.operational_trust.assessor import OperationalTrustAssessor


def complete_reproducibility_metadata():
    """Return complete reproducibility evidence for integration tests."""
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
    """Return complete monitoring evidence for integration tests."""
    return {
        "monitoring_metrics": [
            "data_drift",
            "missingness",
            "prediction_performance",
        ],
        "alerting_rules": {
            "data_drift": "greater_than_threshold",
        },
        "monitoring_owner": "ml-platform",
        "review_cadence": "monthly",
        "logging_enabled": True,
        "incident_process": "operational-runbook",
        "reassessment_triggers": [
            "material_data_change",
            "performance_degradation",
        ],
        "monitoring_history": [
            "2026-09-review",
            "2026-10-review",
        ],
    }


def test_monitoring_runs_without_optional_metadata():
    """Monitoring analysis should run without optional metadata."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    monitoring = report["monitoring_readiness"]

    assert monitoring is not None
    assert monitoring["metadata_supplied"] is False
    assert monitoring["review_required"] is True
    assert monitoring["review_reasons"] == [
        "missing_monitoring_evidence"
    ]

    assert "monitoring_readiness" in (
        report["operational_trust_summary"]["review_reasons"]
    )


def test_complete_monitoring_metadata_removes_monitoring_review():
    """Complete monitoring evidence should remove its review reason."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(
        data,
        monitoring_metadata=complete_monitoring_metadata(),
    ).assess()

    monitoring = report["monitoring_readiness"]

    assert monitoring["metadata_supplied"] is True
    assert monitoring["missing_monitoring_fields"] == []
    assert monitoring["review_required"] is False

    assert "monitoring_readiness" not in (
        report["operational_trust_summary"]["review_reasons"]
    )

    assert "reproducibility" in (
        report["operational_trust_summary"]["review_reasons"]
    )


def test_partial_monitoring_metadata_propagates_review():
    """Partial monitoring evidence should propagate central review."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    monitoring_metadata = {
        "monitoring_metrics": ["data_drift"],
        "monitoring_owner": "data-team",
        "logging_enabled": True,
    }

    report = OperationalTrustAssessor(
        data,
        monitoring_metadata=monitoring_metadata,
    ).assess()

    monitoring = report["monitoring_readiness"]

    assert monitoring["review_required"] is True
    assert "missing_monitoring_evidence" in (
        monitoring["review_reasons"]
    )

    assert "monitoring_readiness" in (
        report["operational_trust_summary"]["review_reasons"]
    )


def test_assessment_context_reflects_monitoring_metadata():
    """Assessment context should expose monitoring metadata presence."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(
        data,
        monitoring_metadata=complete_monitoring_metadata(),
    ).assess()

    assert (
        report["assessment_context"][
            "monitoring_metadata_supplied"
        ]
        is True
    )


def test_monitoring_availability_exposes_optional_context():
    """Availability should identify monitoring metadata as optional."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    assert report["analysis_availability"][
        "monitoring_readiness"
    ] == {
        "available": True,
        "requires": ["current_data"],
        "optional_context": ["monitoring_metadata"],
    }


def test_disabled_logging_propagates_monitoring_review():
    """Explicitly disabled logging should propagate review."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    monitoring_metadata = complete_monitoring_metadata()
    monitoring_metadata["logging_enabled"] = False

    report = OperationalTrustAssessor(
        data,
        monitoring_metadata=monitoring_metadata,
    ).assess()

    monitoring = report["monitoring_readiness"]

    assert monitoring["logging_analysis"] == {
        "provided": True,
        "enabled": False,
        "requires_review": True,
    }

    assert monitoring["review_reasons"] == [
        "logging_configuration"
    ]

    assert "monitoring_readiness" in (
        report["operational_trust_summary"]["review_reasons"]
    )


def test_complete_monitoring_does_not_hide_reproducibility_review():
    """Monitoring evidence should not suppress other review reasons."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(
        data,
        monitoring_metadata=complete_monitoring_metadata(),
    ).assess()

    summary = report["operational_trust_summary"]

    assert summary["review_required"] is True
    assert summary["review_reasons"] == [
        "reproducibility"
    ]


def test_complete_reproducibility_does_not_hide_monitoring_review():
    """Reproducibility evidence should not suppress monitoring review."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
        identifier_column="record_id",
    ).assess()

    summary = report["operational_trust_summary"]

    assert report["reproducibility"]["review_required"] is False
    assert report["monitoring_readiness"]["review_required"] is True

    assert summary["review_reasons"] == [
        "monitoring_readiness"
    ]


def test_complete_reproducibility_and_monitoring_can_clear_review():
    """Complete evidence should allow no-review state when data is stable."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "feature": [10, 20, 30, 40],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
        identifier_column="record_id",
        monitoring_metadata=complete_monitoring_metadata(),
    ).assess()

    assert report["data_drift"] is None
    assert report["data_stability"]["review_required"] is False
    assert report["reproducibility"]["review_required"] is False
    assert report["monitoring_readiness"]["review_required"] is False

    assert report["operational_trust_summary"] == {
        "status": "no_review_indicators",
        "review_required": False,
        "review_reasons": [],
    }


def test_monitoring_coexists_with_drift_review():
    """Monitoring evidence should coexist with independent drift review."""
    reference_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "feature": [10, 10, 10, 10],
        }
    )

    data = pd.DataFrame(
        {
            "record_id": [5, 6, 7, 8],
            "feature": [30, 30, 30, 30],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reference_data=reference_data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
        identifier_column="record_id",
        monitoring_metadata=complete_monitoring_metadata(),
    ).assess()

    assert report["data_drift"]["review_required"] is True
    assert report["monitoring_readiness"]["review_required"] is False

    assert report["operational_trust_summary"][
        "review_reasons"
    ] == [
        "data_drift"
    ]


def test_monitoring_coexists_with_stability_review():
    """Monitoring evidence should coexist with stability review."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "feature": [10, None, None, None],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
        identifier_column="record_id",
        monitoring_metadata=complete_monitoring_metadata(),
    ).assess()

    assert report["data_stability"]["review_required"] is True
    assert report["reproducibility"]["review_required"] is False
    assert report["monitoring_readiness"]["review_required"] is False

    assert report["operational_trust_summary"][
        "review_reasons"
    ] == [
        "data_stability"
    ]


def test_all_operational_review_dimensions_can_coexist():
    """All Operational Trust review dimensions should coexist."""
    reference_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "feature": [10, 10, 10, 10],
        }
    )

    data = pd.DataFrame(
        {
            "record_id": [5, 6, 7, 8],
            "feature": [30, None, None, None],
        }
    )

    monitoring_metadata = {
        "monitoring_metrics": ["data_drift"],
        "logging_enabled": False,
    }

    report = OperationalTrustAssessor(
        data,
        reference_data=reference_data,
        monitoring_metadata=monitoring_metadata,
    ).assess()

    summary = report["operational_trust_summary"]

    assert report["data_drift"]["review_required"] is True
    assert report["data_stability"]["review_required"] is True
    assert report["reproducibility"]["review_required"] is True
    assert report["monitoring_readiness"]["review_required"] is True

    assert summary["review_required"] is True

    assert summary["review_reasons"] == [
        "data_drift",
        "data_stability",
        "reproducibility",
        "monitoring_readiness",
    ]


def test_additional_monitoring_metadata_is_preserved():
    """Additional monitoring evidence should survive integration."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    monitoring_metadata = complete_monitoring_metadata()
    monitoring_metadata["dashboard_url"] = "internal-dashboard"

    report = OperationalTrustAssessor(
        data,
        monitoring_metadata=monitoring_metadata,
    ).assess()

    assert report["monitoring_readiness"][
        "additional_metadata_fields"
    ] == [
        "dashboard_url"
    ]


def test_detailed_monitoring_output_is_preserved():
    """Central assessor should preserve detailed monitoring evidence."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "feature": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(
        data,
        monitoring_metadata=complete_monitoring_metadata(),
    ).assess()

    monitoring = report["monitoring_readiness"]

    assert monitoring["evidence_coverage"] == {
        "recognised_fields": 8,
        "provided_fields": 8,
        "missing_fields": 0,
        "coverage_rate": 1.0,
    }

    assert monitoring["monitoring_evidence"][
        "monitoring_owner"
    ] == {
        "provided": True,
        "value": "ml-platform",
    }

    assert monitoring["monitoring_history_analysis"] == {
        "provided": True,
        "value": [
            "2026-09-review",
            "2026-10-review",
        ],
        "requires_review": False,
    }
