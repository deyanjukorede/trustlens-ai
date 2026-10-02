"""
End-to-end integration tests for TrustLens AI Phase 3 Data Governance.

These tests validate that the governance capabilities developed during
Phase 3 operate together through the DataGovernanceAssessor without
breaking existing assessment behaviour.
"""

import pandas as pd
import pytest

from trustlens.governance.assessor import DataGovernanceAssessor


@pytest.fixture
def governance_dataset():
    """Create a representative dataset for end-to-end governance testing."""
    return pd.DataFrame(
        {
            "customer_id": [1001, 1002, 1003, 1004],
            "full_name": [
                "Ada Smith",
                "John Brown",
                "Mary Jones",
                "David Green",
            ],
            "email": [
                "ada@example.com",
                "john@example.com",
                "mary@example.com",
                "david@example.com",
            ],
            "mobile_number": [
                "+447700900001",
                "+447700900002",
                "+447700900003",
                "+447700900004",
            ],
            "age": [28, 35, 42, 31],
            "income": [50000, 62000, 58000, 71000],
            "risk": ["low", "medium", "high", "low"],
        }
    )


@pytest.fixture
def governance_metadata():
    """Create complete column metadata for the integration dataset."""
    return {
        "customer_id": {
            "description": "Unique customer identifier",
            "owner": "Customer Operations",
        },
        "full_name": {
            "description": "Customer full name",
            "owner": "Customer Operations",
        },
        "email": {
            "description": "Customer email address",
            "owner": "Customer Operations",
        },
        "mobile_number": {
            "description": "Customer mobile telephone number",
            "owner": "Customer Operations",
        },
        "age": {
            "description": "Customer age",
            "owner": "Risk Analytics",
        },
        "income": {
            "description": "Customer annual income",
            "owner": "Risk Analytics",
        },
        "risk": {
            "description": "Customer risk classification",
            "owner": "Risk Analytics",
        },
    }


@pytest.fixture
def governance_controls():
    """Create a complete set of governance controls."""
    return {
        "data_owner": "Customer Operations",
        "approved_purpose": "Customer risk assessment",
        "data_classification": "Confidential",
        "access_control": True,
        "retention_policy": "7 years",
        "review_process": "Annual governance review",
        "accountability": "Head of Data",
    }


def test_phase3_governance_report_contains_core_sections(
    governance_dataset,
    governance_metadata,
    governance_controls,
):
    """Complete Phase 3 assessment should expose all governance sections."""
    assessor = DataGovernanceAssessor(
        governance_dataset,
        metadata=governance_metadata,
        governance_controls=governance_controls,
    )

    report = assessor.assess()

    expected_sections = {
        "dataset",
        "column_inventory",
        "sensitive_data",
        "privacy_risk",
        "metadata_completeness",
        "governance_controls",
    }

    assert expected_sections.issubset(report.keys())


def test_phase3_sensitive_data_flows_into_complete_report(
    governance_dataset,
    governance_metadata,
    governance_controls,
):
    """Sensitive-data analysis should be integrated into the final report."""
    assessor = DataGovernanceAssessor(
        governance_dataset,
        metadata=governance_metadata,
        governance_controls=governance_controls,
    )

    report = assessor.assess()

    assert report["sensitive_data"] is not None
    assert report["privacy_risk"] is not None


def test_phase3_metadata_flows_into_complete_report(
    governance_dataset,
    governance_metadata,
    governance_controls,
):
    """Metadata completeness should be integrated into the final report."""
    assessor = DataGovernanceAssessor(
        governance_dataset,
        metadata=governance_metadata,
        governance_controls=governance_controls,
    )

    report = assessor.assess()

    assert report["metadata_completeness"] is not None


def test_phase3_governance_controls_flow_into_complete_report(
    governance_dataset,
    governance_metadata,
    governance_controls,
):
    """Governance controls should be integrated into the final report."""
    assessor = DataGovernanceAssessor(
        governance_dataset,
        metadata=governance_metadata,
        governance_controls=governance_controls,
    )

    report = assessor.assess()

    controls = report["governance_controls"]

    assert controls is not None
    assert controls["controls_assessed"] == 7
    assert controls["controls_implemented"] == 7
    assert controls["controls_missing"] == 0
    assert controls["governance_controls_score"] == 100.0


def test_phase3_dataset_structure_is_preserved(
    governance_dataset,
    governance_metadata,
    governance_controls,
):
    """Governance integration should preserve structural assessment."""
    assessor = DataGovernanceAssessor(
        governance_dataset,
        metadata=governance_metadata,
        governance_controls=governance_controls,
    )

    report = assessor.assess()

    assert report["dataset"]["rows"] == 4
    assert report["dataset"]["columns"] == 7
    assert set(report["column_inventory"].keys()) == set(
        governance_dataset.columns
    )


def test_phase3_assessment_is_deterministic(
    governance_dataset,
    governance_metadata,
    governance_controls,
):
    """Repeated Phase 3 assessments should produce identical results."""
    assessor = DataGovernanceAssessor(
        governance_dataset,
        metadata=governance_metadata,
        governance_controls=governance_controls,
    )

    first_report = assessor.assess()
    second_report = assessor.assess()

    assert first_report == second_report


def test_phase3_remains_backward_compatible_without_optional_inputs(
    governance_dataset,
):
    """Existing users should still be able to assess a dataset alone."""
    assessor = DataGovernanceAssessor(governance_dataset)

    report = assessor.assess()

    assert report["dataset"]["rows"] == 4
    assert report["dataset"]["columns"] == 7
    assert report["metadata_completeness"] is None
    assert report["governance_controls"] is None
