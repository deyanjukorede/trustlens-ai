import pandas as pd
import pytest

from trustlens.readiness.assessor import AIReadinessAssessor


@pytest.fixture
def sample_data():
    """Create a representative dataset for AI readiness assessment."""
    return pd.DataFrame(
        {
            "age": [25, 31, 42, 36],
            "income": [50000, 62000, 78000, 55000],
            "city": ["Sheffield", "Leeds", "London", "Sheffield"],
            "verified": [True, False, True, True],
            "target": [0, 1, 1, 0],
        }
    )


def test_dataset_dimensions(sample_data):
    """Assessor should correctly identify dataset dimensions."""
    assessor = AIReadinessAssessor(sample_data, target_column="target")

    assert assessor.row_count == 4
    assert assessor.column_count == 5


def test_target_column_is_stored(sample_data):
    """Assessor should retain the supplied target column."""
    assessor = AIReadinessAssessor(sample_data, target_column="target")

    assert assessor.target_column == "target"


def test_feature_columns_exclude_target(sample_data):
    """Target column should not be treated as a modelling feature."""
    assessor = AIReadinessAssessor(sample_data, target_column="target")

    assert assessor.feature_columns == [
        "age",
        "income",
        "city",
        "verified",
    ]


def test_feature_columns_without_target(sample_data):
    """All columns should be features when no target is supplied."""
    assessor = AIReadinessAssessor(sample_data)

    assert assessor.feature_columns == [
        "age",
        "income",
        "city",
        "verified",
        "target",
    ]


def test_feature_type_summary(sample_data):
    """Assessor should classify modelling features by broad data type."""
    assessor = AIReadinessAssessor(sample_data, target_column="target")
    summary = assessor.feature_type_summary()

    assert set(summary["numeric"]) == {"age", "income"}
    assert summary["categorical"] == ["city"]
    assert summary["boolean"] == ["verified"]
    assert summary["datetime"] == []


def test_missing_value_summary():
    """Assessor should correctly summarise missing values."""
    data = pd.DataFrame(
        {
            "age": [25, None, 42, 36],
            "income": [50000, 62000, None, 55000],
            "target": [0, 1, 1, 0],
        }
    )

    assessor = AIReadinessAssessor(data, target_column="target")
    summary = assessor.missing_value_summary()

    assert summary["total_missing_values"] == 2
    assert summary["missing_rate"] == pytest.approx(16.67, abs=0.01)
    assert set(summary["columns_with_missing_values"]) == {"age", "income"}
    assert summary["missing_by_column"]["age"] == 1
    assert summary["missing_by_column"]["income"] == 1
    assert summary["missing_by_column"]["target"] == 0


def test_assess_returns_expected_sections(sample_data):
    """Full readiness assessment should expose foundational sections."""
    assessor = AIReadinessAssessor(sample_data, target_column="target")
    report = assessor.assess()

    assert "dataset" in report
    assert "target_column" in report
    assert "feature_count" in report
    assert "feature_columns" in report
    assert "feature_types" in report
    assert "missing_values" in report


def test_assess_dataset_information(sample_data):
    """Assessment should contain correct dataset information."""
    assessor = AIReadinessAssessor(sample_data, target_column="target")
    report = assessor.assess()

    assert report["dataset"]["rows"] == 4
    assert report["dataset"]["columns"] == 5
    assert report["target_column"] == "target"
    assert report["feature_count"] == 4


def test_assess_feature_columns(sample_data):
    """Assessment should expose the modelling feature list."""
    assessor = AIReadinessAssessor(sample_data, target_column="target")
    report = assessor.assess()

    assert report["feature_columns"] == [
        "age",
        "income",
        "city",
        "verified",
    ]


def test_invalid_input():
    """Assessor should reject input that is not a pandas DataFrame."""
    with pytest.raises(TypeError):
        AIReadinessAssessor([1, 2, 3])


def test_empty_dataframe():
    """Assessor should reject an empty pandas DataFrame."""
    with pytest.raises(ValueError):
        AIReadinessAssessor(pd.DataFrame())


def test_invalid_target_column(sample_data):
    """Assessor should reject a target column that does not exist."""
    with pytest.raises(ValueError):
        AIReadinessAssessor(
            sample_data,
            target_column="nonexistent_target",
        )


def test_single_feature_with_target():
    """Assessor should work with one feature and one target column."""
    data = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "target": [0, 1, 0],
        }
    )

    assessor = AIReadinessAssessor(data, target_column="target")
    report = assessor.assess()

    assert report["dataset"]["rows"] == 3
    assert report["dataset"]["columns"] == 2
    assert report["feature_count"] == 1
    assert report["feature_columns"] == ["age"]


def test_datetime_feature_classification():
    """Datetime columns should be classified separately."""
    data = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"]
            ),
            "target": [0, 1, 0],
        }
    )

    assessor = AIReadinessAssessor(data, target_column="target")
    summary = assessor.feature_type_summary()

    assert summary["datetime"] == ["event_date"]
    assert summary["numeric"] == []
    assert summary["categorical"] == []
    assert summary["boolean"] == []
