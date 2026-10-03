import pandas as pd

from trustlens.operational_trust.assessor import OperationalTrustAssessor


def complete_reproducibility_metadata():
    """Return complete reproducibility evidence for integration tests."""
    return {
        "dataset_version": "dataset-v1",
        "code_version": "commit-abc123",
        "random_seed": 42,
        "environment": "python-3.11",
        "data_source": "warehouse",
        "lineage": "raw -> curated -> model",
        "execution_id": "run-001",
    }


def test_reproducibility_runs_without_optional_metadata():
    """Reproducibility analysis should run with current data alone."""
    data = pd.DataFrame(
        {
            "score": [10, 20, 30, 40],
            "segment": ["A", "B", "A", "B"],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    reproducibility = report["reproducibility"]

    assert reproducibility is not None
    assert reproducibility["metadata_supplied"] is False
    assert reproducibility["review_required"] is True

    assert reproducibility["review_reasons"] == [
        "missing_reproducibility_metadata"
    ]

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": [
            "reproducibility",
            "monitoring_readiness",
        ],
    }


def test_complete_metadata_removes_reproducibility_review_reason():
    """Complete evidence should avoid a reproducibility review reason."""
    data = pd.DataFrame(
        {
            "score": [10, 20, 30, 40],
            "segment": ["A", "B", "A", "B"],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
    ).assess()

    reproducibility = report["reproducibility"]

    assert reproducibility["metadata_supplied"] is True
    assert reproducibility["missing_metadata_fields"] == []
    assert reproducibility["review_required"] is False

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": ["monitoring_readiness"],
    }


def test_partial_metadata_propagates_reproducibility_review():
    """Missing reproducibility evidence should propagate centrally."""
    data = pd.DataFrame(
        {
            "score": [10, 20, 30, 40],
            "segment": ["A", "B", "A", "B"],
        }
    )

    metadata = {
        "dataset_version": "dataset-v2",
        "code_version": "commit-123",
        "random_seed": 7,
    }

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=metadata,
    ).assess()

    reproducibility = report["reproducibility"]

    assert reproducibility["metadata_coverage"]["provided_fields"] == 3
    assert reproducibility["metadata_coverage"]["missing_fields"] == 4

    assert reproducibility["review_required"] is True

    assert (
        "missing_reproducibility_metadata"
        in reproducibility["review_reasons"]
    )

    assert (
        "reproducibility"
        in report["operational_trust_summary"]["review_reasons"]
    )


def test_reproducibility_metadata_is_reflected_in_context():
    """Assessment context should report supplied reproducibility evidence."""
    data = pd.DataFrame(
        {
            "score": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
    ).assess()

    context = report["assessment_context"]

    assert context["reproducibility_metadata_supplied"] is True
    assert context["identifier_column"] is None


def test_identifier_column_is_passed_to_reproducibility_analysis():
    """Identifier configuration should flow through the assessor."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3", "R4"],
            "score": [10, 20, 30, 40],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
        identifier_column="record_id",
    ).assess()

    identifier = report["reproducibility"]["identifier_analysis"]

    assert identifier["provided"] is True
    assert identifier["column"] == "record_id"
    assert identifier["missing_values"] == 0
    assert identifier["duplicate_values"] == 0
    assert identifier["is_complete"] is True
    assert identifier["is_unique"] is True
    assert identifier["requires_review"] is False

    assert (
        report["assessment_context"]["identifier_column"]
        == "record_id"
    )


def test_identifier_integrity_issue_propagates_to_summary():
    """Identifier integrity findings should propagate centrally."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", "R1", "R2", "R3"],
            "score": [10, 20, 30, 40],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
        identifier_column="record_id",
    ).assess()

    reproducibility = report["reproducibility"]

    assert (
        reproducibility["identifier_analysis"]["requires_review"]
        is True
    )

    assert reproducibility["review_reasons"] == [
        "identifier_integrity"
    ]

    assert (
        "reproducibility"
        in report["operational_trust_summary"]["review_reasons"]
    )


def test_missing_metadata_and_identifier_issue_coexist():
    """Independent reproducibility indicators should remain visible."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", "R1", None, "R3"],
            "score": [10, 20, 30, 40],
        }
    )

    report = OperationalTrustAssessor(
        data,
        identifier_column="record_id",
    ).assess()

    reproducibility = report["reproducibility"]

    assert reproducibility["review_reasons"] == [
        "missing_reproducibility_metadata",
        "identifier_integrity",
    ]

    assert reproducibility["review_required"] is True

    assert (
        "reproducibility"
        in report["operational_trust_summary"]["review_reasons"]
    )


def test_reproducibility_availability_exposes_optional_context():
    """Availability should describe optional reproducibility evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    availability = OperationalTrustAssessor(
        data
    ).analysis_availability()

    assert availability["reproducibility"] == {
        "available": True,
        "requires": ["current_data"],
        "optional_context": [
            "reproducibility_metadata",
            "identifier_column",
        ],
    }


def test_additional_metadata_is_preserved_through_integration():
    """Additional workflow metadata should survive integration."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = complete_reproducibility_metadata()
    metadata["model_version"] = "model-v5"
    metadata["owner"] = "ml-platform"

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=metadata,
    ).assess()

    assert report["reproducibility"][
        "additional_metadata_fields"
    ] == [
        "model_version",
        "owner",
    ]


def test_drift_and_reproducibility_review_reasons_can_coexist():
    """Drift and reproducibility findings should coexist."""
    reference_data = pd.DataFrame(
        {
            "score": [10, 11, 12, 13],
        }
    )

    current_data = pd.DataFrame(
        {
            "score": [20, 21, 22, 23],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
    ).assess()

    assert report["data_drift"]["review_required"] is True
    assert report["data_stability"]["review_required"] is False
    assert report["reproducibility"]["review_required"] is True

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": [
            "data_drift",
            "reproducibility",
            "monitoring_readiness",
        ],
    }


def test_stability_and_reproducibility_review_reasons_can_coexist():
    """Stability and reproducibility findings should coexist."""
    data = pd.DataFrame(
        {
            "constant": [7, 7, 7, 7],
            "variable": [1, 2, 3, 4],
        }
    )

    report = OperationalTrustAssessor(data).assess()

    assert report["data_stability"]["review_required"] is True
    assert report["reproducibility"]["review_required"] is True

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": [
            "data_stability",
            "reproducibility",
            "monitoring_readiness",
        ],
    }


def test_all_three_operational_components_can_require_review():
    """Drift, stability, reproducibility, and monitoring can coexist."""
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

    assert report["data_drift"]["review_required"] is True
    assert report["data_stability"]["review_required"] is True
    assert report["reproducibility"]["review_required"] is True
    assert report["monitoring_readiness"]["review_required"] is True

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": [
            "data_drift",
            "data_stability",
            "reproducibility",
            "monitoring_readiness",
        ],
    }


def test_complete_reproducibility_evidence_does_not_hide_other_findings():
    """Complete reproducibility evidence should not suppress other findings."""
    reference_data = pd.DataFrame(
        {
            "score": [10, 11, 12, 13],
        }
    )

    current_data = pd.DataFrame(
        {
            "score": [20, 21, 22, 23],
        }
    )

    report = OperationalTrustAssessor(
        current_data,
        reference_data=reference_data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
    ).assess()

    assert report["data_drift"]["review_required"] is True
    assert report["reproducibility"]["review_required"] is False
    assert report["monitoring_readiness"]["review_required"] is True

    assert report["operational_trust_summary"] == {
        "status": "review",
        "review_required": True,
        "review_reasons": [
            "data_drift",
            "monitoring_readiness",
        ],
    }


def test_reproducibility_detail_is_preserved_in_integrated_report():
    """Central integration should preserve reproducibility evidence."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3"],
            "score": [10, 20, 30],
        }
    )

    report = OperationalTrustAssessor(
        data,
        reproducibility_metadata=(
            complete_reproducibility_metadata()
        ),
        identifier_column="record_id",
    ).assess()

    reproducibility = report["reproducibility"]

    assert "metadata_evidence" in reproducibility
    assert "metadata_coverage" in reproducibility
    assert "missing_metadata_fields" in reproducibility
    assert "identifier_analysis" in reproducibility
    assert "review_required" in reproducibility
    assert "review_reasons" in reproducibility

    assert reproducibility["metadata_coverage"]["coverage_rate"] == 1.0
