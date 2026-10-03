import pandas as pd
import pytest

from trustlens.operational_trust.monitoring import (
    MonitoringReadinessAnalyzer,
)


def complete_monitoring_metadata():
    """Return complete monitoring evidence for tests."""
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


def test_analyzer_initialises_with_valid_dataframe():
    """Analyzer should accept a non-empty DataFrame."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    analyzer = MonitoringReadinessAnalyzer(data)

    assert analyzer.row_count == 3
    assert analyzer.column_count == 1
    assert analyzer.metadata == {}


def test_analyzer_rejects_non_dataframe_input():
    """Analyzer should reject non-DataFrame input."""
    with pytest.raises(TypeError):
        MonitoringReadinessAnalyzer(
            [1, 2, 3]
        )


def test_analyzer_rejects_empty_dataframe():
    """Analyzer should reject an empty DataFrame."""
    with pytest.raises(ValueError):
        MonitoringReadinessAnalyzer(
            pd.DataFrame()
        )


def test_analyzer_rejects_non_dictionary_metadata():
    """Monitoring metadata should be supplied as a dictionary."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(TypeError):
        MonitoringReadinessAnalyzer(
            data,
            metadata=["monitoring"],
        )


def test_missing_metadata_is_reported():
    """Missing monitoring evidence should be explicitly reported."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    report = MonitoringReadinessAnalyzer(data).assess()

    assert report["metadata_supplied"] is False

    assert report["missing_monitoring_fields"] == [
        "monitoring_metrics",
        "alerting_rules",
        "monitoring_owner",
        "review_cadence",
        "logging_enabled",
        "incident_process",
        "reassessment_triggers",
        "monitoring_history",
    ]

    assert report["evidence_coverage"] == {
        "recognised_fields": 8,
        "provided_fields": 0,
        "missing_fields": 8,
        "coverage_rate": 0.0,
    }

    assert report["review_required"] is True

    assert report["review_reasons"] == [
        "missing_monitoring_evidence"
    ]


def test_complete_metadata_produces_full_evidence_coverage():
    """Complete monitoring metadata should provide full coverage."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=complete_monitoring_metadata(),
    ).assess()

    assert report["metadata_supplied"] is True
    assert report["missing_monitoring_fields"] == []

    assert report["evidence_coverage"] == {
        "recognised_fields": 8,
        "provided_fields": 8,
        "missing_fields": 0,
        "coverage_rate": 1.0,
    }

    assert report["review_required"] is False
    assert report["review_reasons"] == []


def test_partial_metadata_reports_missing_evidence():
    """Partial metadata should identify remaining evidence gaps."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "monitoring_metrics": ["data_drift"],
        "monitoring_owner": "data-team",
        "review_cadence": "monthly",
        "logging_enabled": True,
    }

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["evidence_coverage"]["provided_fields"] == 4
    assert report["evidence_coverage"]["missing_fields"] == 4
    assert report["evidence_coverage"]["coverage_rate"] == 0.5

    assert report["missing_monitoring_fields"] == [
        "alerting_rules",
        "incident_process",
        "reassessment_triggers",
        "monitoring_history",
    ]

    assert report["review_required"] is True

    assert report["review_reasons"] == [
        "missing_monitoring_evidence"
    ]


def test_blank_string_does_not_count_as_monitoring_evidence():
    """Blank strings should not count as monitoring evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()
    metadata["monitoring_owner"] = "   "

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["monitoring_evidence"][
        "monitoring_owner"
    ]["provided"] is False

    assert "monitoring_owner" in (
        report["missing_monitoring_fields"]
    )


def test_empty_collection_does_not_count_as_monitoring_evidence():
    """Empty collections should not count as evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()
    metadata["monitoring_metrics"] = []

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["monitoring_evidence"][
        "monitoring_metrics"
    ]["provided"] is False

    assert "monitoring_metrics" in (
        report["missing_monitoring_fields"]
    )


def test_logging_true_is_supplied_enabled_evidence():
    """Explicit True logging evidence should be recognised."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["logging_analysis"] == {
        "provided": True,
        "enabled": True,
        "requires_review": False,
    }


def test_logging_false_is_distinct_review_indicator():
    """Explicitly disabled logging should require review."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()
    metadata["logging_enabled"] = False

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["monitoring_evidence"][
        "logging_enabled"
    ]["provided"] is True

    assert report["logging_analysis"] == {
        "provided": True,
        "enabled": False,
        "requires_review": True,
    }

    assert report["missing_monitoring_fields"] == []

    assert report["review_required"] is True

    assert report["review_reasons"] == [
        "logging_configuration"
    ]


def test_missing_logging_is_evidence_gap_not_disabled_control():
    """Missing logging evidence should differ from disabled logging."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()
    del metadata["logging_enabled"]

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["logging_analysis"] == {
        "provided": False,
        "enabled": None,
        "requires_review": False,
    }

    assert "logging_enabled" in (
        report["missing_monitoring_fields"]
    )

    assert report["review_reasons"] == [
        "missing_monitoring_evidence"
    ]


def test_non_boolean_logging_value_requires_review():
    """Non-boolean logging configuration should require review."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()
    metadata["logging_enabled"] = "yes"

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["logging_analysis"] == {
        "provided": True,
        "enabled": None,
        "requires_review": True,
    }

    assert report["review_required"] is True

    assert report["review_reasons"] == [
        "logging_configuration"
    ]


def test_missing_evidence_and_logging_issue_can_coexist():
    """Independent monitoring review indicators should coexist."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "monitoring_metrics": ["data_drift"],
        "logging_enabled": False,
    }

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["review_required"] is True

    assert report["review_reasons"] == [
        "missing_monitoring_evidence",
        "logging_configuration",
    ]


def test_monitoring_history_is_reported_when_supplied():
    """Monitoring history evidence should be preserved."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["monitoring_history_analysis"] == {
        "provided": True,
        "value": [
            "2026-09-review",
            "2026-10-review",
        ],
        "requires_review": False,
    }


def test_missing_monitoring_history_is_reported():
    """Absent monitoring history should appear as missing evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()
    del metadata["monitoring_history"]

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["monitoring_history_analysis"] == {
        "provided": False,
        "value": None,
        "requires_review": False,
    }

    assert "monitoring_history" in (
        report["missing_monitoring_fields"]
    )


def test_additional_metadata_fields_are_preserved():
    """Additional monitoring metadata should remain visible."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_monitoring_metadata()
    metadata["dashboard_url"] = "internal-dashboard"
    metadata["monitoring_tool"] = "observability-platform"

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert report["additional_metadata_fields"] == [
        "dashboard_url",
        "monitoring_tool",
    ]


def test_assessment_returns_expected_top_level_structure():
    """Assessment should expose the monitoring evidence contract."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    report = MonitoringReadinessAnalyzer(
        data,
        metadata=complete_monitoring_metadata(),
    ).assess()

    assert set(report.keys()) == {
        "dataset",
        "metadata_supplied",
        "monitoring_evidence",
        "evidence_coverage",
        "missing_monitoring_fields",
        "additional_metadata_fields",
        "logging_analysis",
        "monitoring_history_analysis",
        "review_required",
        "review_reasons",
    }

    assert report["dataset"] == {
        "rows": 3,
        "columns": 1,
    }
