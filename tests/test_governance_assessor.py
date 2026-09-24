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
