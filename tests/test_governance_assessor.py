import pandas as pd
import pytest

from trustlens.governance.assessor import DataGovernanceAssessor


@pytest.fixture
def sample_data():
    """Create a representative dataset for governance assessment."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "age": [25, 31, None, 40],
            "income": [50000, 62000, 58000, 58000],
            "risk": ["low", "medium", "high", "high"],
        }
    )


@pytest.fixture
def sensitive_governance_data():
    """Create a dataset containing sensitive information."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "full_name": [
                "Ada Smith",
                "John Brown",
                "Mary Jones",
            ],
            "email": [
                "ada@example.com",
                "john@example.com",
                "mary@example.com",
            ],
            "mobile_number": [
                "+447700900001",
                "+447700900002",
                "+447700900003",
            ],
            "age": [28, 35, 42],
        }
    )


def test_dataset_dimensions(sample_data):
    """Assessor should correctly identify dataset dimensions."""
    assessor = DataGovernanceAssessor(sample_data)

    assert assessor.row_count == 4
    assert assessor.column_count == 4


def test_column_inventory_contains_all_columns(sample_data):
    """Column inventory should contain every dataset column."""
    assessor = DataGovernanceAssessor(sample_data)
    inventory = assessor.column_inventory()

    assert set(inventory.keys()) == {
        "customer_id",
        "age",
        "income",
        "risk",
    }


def test_column_inventory_structure(sample_data):
    """Each inventory entry should contain foundational metadata."""
    assessor = DataGovernanceAssessor(sample_data)
    inventory = assessor.column_inventory()

    for column in sample_data.columns:
        assert "data_type" in inventory[column]
        assert "missing_count" in inventory[column]
        assert "unique_count" in inventory[column]


def test_missing_value_count(sample_data):
    """Assessor should correctly count missing values by column."""
    assessor = DataGovernanceAssessor(sample_data)
    inventory = assessor.column_inventory()

    assert inventory["customer_id"]["missing_count"] == 0
    assert inventory["age"]["missing_count"] == 1
    assert inventory["income"]["missing_count"] == 0
    assert inventory["risk"]["missing_count"] == 0


def test_unique_value_count(sample_data):
    """Assessor should correctly count unique non-missing values."""
    assessor = DataGovernanceAssessor(sample_data)
    inventory = assessor.column_inventory()

    assert inventory["customer_id"]["unique_count"] == 4
    assert inventory["age"]["unique_count"] == 3
    assert inventory["income"]["unique_count"] == 3
    assert inventory["risk"]["unique_count"] == 3


def test_data_types_are_reported(sample_data):
    """Assessor should report pandas data types as strings."""
    assessor = DataGovernanceAssessor(sample_data)
    inventory = assessor.column_inventory()

    assert inventory["customer_id"]["data_type"] == "int64"
    assert inventory["age"]["data_type"] == "float64"
    assert inventory["income"]["data_type"] == "int64"

    # Pandas may represent text columns as either "object" or "str"
    # depending on the pandas/Python environment.
    assert inventory["risk"]["data_type"] in {"object", "str"}


def test_assess_returns_expected_sections(sample_data):
    """Full governance assessment should return expected sections."""
    assessor = DataGovernanceAssessor(sample_data)
    report = assessor.assess()

    assert "dataset" in report
    assert "column_inventory" in report
    assert "sensitive_data" in report


def test_assess_dataset_information(sample_data):
    """Full assessment should contain correct dataset information."""
    assessor = DataGovernanceAssessor(sample_data)
    report = assessor.assess()

    assert report["dataset"]["rows"] == 4
    assert report["dataset"]["columns"] == 4


def test_assess_contains_column_inventory(sample_data):
    """Full assessment should include the generated column inventory."""
    assessor = DataGovernanceAssessor(sample_data)
    report = assessor.assess()

    assert report["column_inventory"] == assessor.column_inventory()


def test_sensitive_data_analysis_structure(sensitive_governance_data):
    """Sensitive-data analysis should return the expected structure."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    analysis = assessor.sensitive_data_analysis()

    assert "sensitive_columns_detected" in analysis
    assert "sensitive_columns" in analysis
    assert "sensitive_types_detected" in analysis
    assert "details" in analysis


def test_sensitive_data_analysis_detects_sensitive_columns(
    sensitive_governance_data,
):
    """Governance assessor should identify sensitive dataset columns."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    analysis = assessor.sensitive_data_analysis()

    assert "full_name" in analysis["sensitive_columns"]
    assert "email" in analysis["sensitive_columns"]
    assert "mobile_number" in analysis["sensitive_columns"]

    assert analysis["sensitive_columns_detected"] >= 3


def test_sensitive_data_analysis_detects_sensitive_types(
    sensitive_governance_data,
):
    """Governance assessor should report detected sensitive-data types."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    analysis = assessor.sensitive_data_analysis()

    assert "name" in analysis["sensitive_types_detected"]
    assert "email" in analysis["sensitive_types_detected"]
    assert "phone_number" in analysis["sensitive_types_detected"]


def test_assess_contains_sensitive_data_analysis(
    sensitive_governance_data,
):
    """Full governance report should contain sensitive-data analysis."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    report = assessor.assess()

    assert "sensitive_data" in report
    assert report["sensitive_data"] == assessor.sensitive_data_analysis()


def test_assess_sensitive_data_details(
    sensitive_governance_data,
):
    """Full report should expose details of detected sensitive columns."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    report = assessor.assess()

    details = report["sensitive_data"]["details"]

    assert "full_name" in details
    assert "email" in details
    assert "mobile_number" in details

    assert "name" in details["full_name"]["sensitive_types"]
    assert "email" in details["email"]["sensitive_types"]
    assert "phone_number" in details["mobile_number"]["sensitive_types"]


def test_clean_data_has_no_sensitive_value_patterns():
    """Ordinary business data should not create sensitive value matches."""
    data = pd.DataFrame(
        {
            "product": [
                "Laptop",
                "Monitor",
                "Keyboard",
            ],
            "quantity": [2, 5, 10],
            "category": [
                "hardware",
                "hardware",
                "accessory",
            ],
        }
    )

    assessor = DataGovernanceAssessor(data)
    analysis = assessor.sensitive_data_analysis()

    assert analysis["sensitive_columns_detected"] == 0
    assert analysis["sensitive_columns"] == []
    assert analysis["sensitive_types_detected"] == []
    assert analysis["details"] == {}


def test_invalid_input():
    """Assessor should reject input that is not a pandas DataFrame."""
    with pytest.raises(TypeError):
        DataGovernanceAssessor([1, 2, 3])


def test_empty_dataframe():
    """Assessor should reject an empty pandas DataFrame."""
    with pytest.raises(ValueError):
        DataGovernanceAssessor(pd.DataFrame())


def test_single_column_dataset():
    """Assessor should work correctly with a single-column dataset."""
    data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
        }
    )

    assessor = DataGovernanceAssessor(data)
    report = assessor.assess()

    assert report["dataset"]["rows"] == 3
    assert report["dataset"]["columns"] == 1
    assert "customer_id" in report["column_inventory"]
    assert report["column_inventory"]["customer_id"]["unique_count"] == 3
    assert "sensitive_data" in report


def test_privacy_risk_analysis_returns_expected_structure(
    sensitive_governance_data,
):
    """Privacy-risk analysis should return the expected assessment structure."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    analysis = assessor.privacy_risk_analysis()

    assert isinstance(analysis, dict)
    assert "privacy_risk_score" in analysis
    assert "risk_level" in analysis
    assert "sensitive_columns_detected" in analysis
    assert "sensitive_column_ratio" in analysis
    assert "sensitive_types_detected" in analysis
    assert "type_risk_score" in analysis
    assert "exposure_score" in analysis
    assert "recommendations" in analysis
    assert "sensitive_data" in analysis


def test_privacy_risk_analysis_detects_sensitive_columns(
    sensitive_governance_data,
):
    """Privacy-risk analysis should recognise sensitive dataset columns."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    analysis = assessor.privacy_risk_analysis()

    assert analysis["sensitive_columns_detected"] > 0
    assert analysis["sensitive_column_ratio"] > 0
    assert len(analysis["sensitive_types_detected"]) > 0


def test_privacy_risk_score_is_valid(
    sensitive_governance_data,
):
    """Privacy-risk score should remain within the supported range."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    analysis = assessor.privacy_risk_analysis()

    assert 0 <= analysis["privacy_risk_score"] <= 100


def test_privacy_risk_level_is_valid(
    sensitive_governance_data,
):
    """Privacy-risk level should use a recognised classification."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    analysis = assessor.privacy_risk_analysis()

    assert analysis["risk_level"] in {
        "low",
        "moderate",
        "high",
        "critical",
    }


def test_privacy_risk_recommendations_are_returned(
    sensitive_governance_data,
):
    """Privacy-risk analysis should provide governance recommendations."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    analysis = assessor.privacy_risk_analysis()

    assert isinstance(analysis["recommendations"], list)
    assert len(analysis["recommendations"]) > 0


def test_assess_contains_privacy_risk(
    sensitive_governance_data,
):
    """Complete governance assessment should include privacy-risk analysis."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    report = assessor.assess()

    assert "privacy_risk" in report
    assert report["privacy_risk"] == assessor.privacy_risk_analysis()


def test_assess_privacy_risk_contains_sensitive_data(
    sensitive_governance_data,
):
    """Integrated privacy-risk report should retain sensitive-data evidence."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    report = assessor.assess()

    assert "sensitive_data" in report["privacy_risk"]
    assert isinstance(report["privacy_risk"]["sensitive_data"], dict)


def test_privacy_risk_integration_preserves_existing_sections(
    sensitive_governance_data,
):
    """Adding privacy risk should preserve existing governance sections."""
    assessor = DataGovernanceAssessor(sensitive_governance_data)
    report = assessor.assess()

    assert "dataset" in report
    assert "column_inventory" in report
    assert "sensitive_data" in report
    assert "privacy_risk" in report
def test_metadata_completeness_analysis_without_metadata(sample_data):
    """Metadata completeness should be None when metadata is not supplied."""
    assessor = DataGovernanceAssessor(sample_data)

    analysis = assessor.metadata_completeness_analysis()

    assert analysis is None


def test_metadata_completeness_analysis_with_complete_metadata(sample_data):
    """Metadata completeness should assess supplied complete metadata."""
    metadata = {
        "customer_id": {
            "description": "Unique customer identifier",
            "owner": "Customer Data Team",
            "business_definition": "Identifier assigned to each customer",
        },
        "age": {
            "description": "Customer age",
            "owner": "Customer Data Team",
            "business_definition": "Age of the customer in years",
        },
        "income": {
            "description": "Customer income",
            "owner": "Finance Data Team",
            "business_definition": "Recorded customer income",
        },
        "risk": {
            "description": "Customer risk category",
            "owner": "Risk Team",
            "business_definition": "Assigned customer risk classification",
        },
    }

    assessor = DataGovernanceAssessor(sample_data, metadata=metadata)
    analysis = assessor.metadata_completeness_analysis()

    assert analysis is not None
    assert analysis["metadata_completeness_score"] == 100
    assert analysis["columns_assessed"] == 4
    assert analysis["columns_with_complete_metadata"] == 4
    assert analysis["columns_with_incomplete_metadata"] == 0


def test_metadata_completeness_analysis_detects_incomplete_metadata(sample_data):
    """Metadata completeness should identify incomplete metadata."""
    metadata = {
        "customer_id": {
            "description": "Unique customer identifier",
            "owner": "Customer Data Team",
            "business_definition": "Identifier assigned to each customer",
        },
        "age": {
            "description": "Customer age",
            "owner": "",
            "business_definition": "Age of the customer in years",
        },
    }

    assessor = DataGovernanceAssessor(sample_data, metadata=metadata)
    analysis = assessor.metadata_completeness_analysis()

    assert analysis is not None
    assert analysis["metadata_completeness_score"] < 100
    assert analysis["columns_with_incomplete_metadata"] > 0
    assert len(analysis["recommendations"]) > 0


def test_assess_contains_metadata_completeness_when_metadata_supplied(sample_data):
    """Complete governance assessment should include metadata completeness."""
    metadata = {
        "customer_id": {
            "description": "Unique customer identifier",
            "owner": "Customer Data Team",
            "business_definition": "Identifier assigned to each customer",
        },
        "age": {
            "description": "Customer age",
            "owner": "Customer Data Team",
            "business_definition": "Age of the customer in years",
        },
        "income": {
            "description": "Customer income",
            "owner": "Finance Data Team",
            "business_definition": "Recorded customer income",
        },
        "risk": {
            "description": "Customer risk category",
            "owner": "Risk Team",
            "business_definition": "Assigned customer risk classification",
        },
    }

    assessor = DataGovernanceAssessor(sample_data, metadata=metadata)
    report = assessor.assess()

    assert "metadata_completeness" in report
    assert report["metadata_completeness"] is not None
    assert report["metadata_completeness"]["metadata_completeness_score"] == 100


def test_assess_metadata_completeness_is_none_without_metadata(sample_data):
    """Governance assessment should remain compatible without metadata."""
    assessor = DataGovernanceAssessor(sample_data)
    report = assessor.assess()

    assert "metadata_completeness" in report
    assert report["metadata_completeness"] is None


def test_metadata_integration_preserves_existing_governance_sections(sample_data):
    """Metadata integration should preserve all existing governance sections."""
    metadata = {
        "customer_id": {
            "description": "Unique customer identifier",
            "owner": "Customer Data Team",
            "business_definition": "Identifier assigned to each customer",
        },
    }

    assessor = DataGovernanceAssessor(sample_data, metadata=metadata)
    report = assessor.assess()

    assert "dataset" in report
    assert "column_inventory" in report
    assert "sensitive_data" in report
    assert "privacy_risk" in report
    assert "metadata_completeness" in report


def test_governance_controls_analysis_without_controls(sample_data):
    """Governance-controls analysis should be None when controls are not supplied."""
    assessor = DataGovernanceAssessor(sample_data)

    analysis = assessor.governance_controls_analysis()

    assert analysis is None


def test_governance_controls_analysis_with_complete_controls(sample_data):
    """Governance-controls analysis should assess supplied controls."""
    governance_controls = {
        "data_owner": "Customer Operations",
        "approved_purpose": "Customer risk assessment",
        "data_classification": "Confidential",
        "access_control": True,
        "retention_policy": "7 years",
        "review_process": "Annual governance review",
        "accountability": "Head of Data",
    }

    assessor = DataGovernanceAssessor(
        sample_data,
        governance_controls=governance_controls,
    )
    analysis = assessor.governance_controls_analysis()

    assert analysis is not None
    assert analysis["controls_assessed"] == 7
    assert analysis["controls_implemented"] == 7
    assert analysis["controls_missing"] == 0
    assert analysis["governance_control_score"] == 100.0


def test_governance_controls_analysis_with_partial_controls(sample_data):
    """Governance-controls analysis should identify missing controls."""
    governance_controls = {
        "data_owner": "Customer Operations",
        "approved_purpose": "Customer risk assessment",
        "data_classification": "Confidential",
        "access_control": True,
    }

    assessor = DataGovernanceAssessor(
        sample_data,
        governance_controls=governance_controls,
    )
    analysis = assessor.governance_controls_analysis()

    assert analysis is not None
    assert analysis["controls_assessed"] == 7
    assert analysis["controls_implemented"] == 4
    assert analysis["controls_missing"] == 3
    assert analysis["governance_control_score"] == 57.14


def test_assess_contains_governance_controls_when_supplied(sample_data):
    """Complete governance assessment should include governance controls."""
    governance_controls = {
        "data_owner": "Customer Operations",
        "approved_purpose": "Customer risk assessment",
        "data_classification": "Confidential",
        "access_control": True,
        "retention_policy": "7 years",
        "review_process": "Annual governance review",
        "accountability": "Head of Data",
    }

    assessor = DataGovernanceAssessor(
        sample_data,
        governance_controls=governance_controls,
    )
    report = assessor.assess()

    assert "governance_controls" in report
    assert report["governance_controls"] is not None
    assert (
        report["governance_controls"]
        == assessor.governance_controls_analysis()
    )


def test_assess_governance_controls_is_none_without_controls(sample_data):
    """Governance assessment should remain compatible without controls."""
    assessor = DataGovernanceAssessor(sample_data)
    report = assessor.assess()

    assert "governance_controls" in report
    assert report["governance_controls"] is None


def test_governance_controls_integration_preserves_existing_sections(
    sample_data,
):
    """Governance-controls integration should preserve existing report sections."""
    governance_controls = {
        "data_owner": "Customer Operations",
        "approved_purpose": "Customer risk assessment",
        "data_classification": "Confidential",
        "access_control": True,
        "retention_policy": "7 years",
        "review_process": "Annual governance review",
        "accountability": "Head of Data",
    }

    assessor = DataGovernanceAssessor(
        sample_data,
        governance_controls=governance_controls,
    )
    report = assessor.assess()

    assert "dataset" in report
    assert "column_inventory" in report
    assert "sensitive_data" in report
    assert "privacy_risk" in report
    assert "metadata_completeness" in report
    assert "governance_controls" in report
