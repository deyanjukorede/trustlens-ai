import pandas as pd
import pytest

from trustlens.quality import DataQualityProfiler


@pytest.fixture
def sample_data():
    """Create a deliberately imperfect dataset for testing."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 3],
            "age": [25, 31, None, 40],
            "income": [50000, 62000, 58000, 58000],
            "risk": ["low", "medium", "high", "high"],
        }
    )


def test_dataset_dimensions(sample_data):
    profiler = DataQualityProfiler(sample_data)

    assert profiler.row_count == 4
    assert profiler.column_count == 4
    assert profiler.total_cells == 16


def test_missing_values(sample_data):
    profiler = DataQualityProfiler(sample_data)
    result = profiler.missing_values()

    assert result["total_missing"] == 1
    assert result["missing_rate"] == 6.25
    assert result["by_column"]["age"]["missing_count"] == 1
    assert result["by_column"]["age"]["missing_rate"] == 25.0


def test_duplicate_rows():
    data = pd.DataFrame(
        {
            "customer_id": [1, 2, 2],
            "risk": ["low", "high", "high"],
        }
    )

    profiler = DataQualityProfiler(data)
    result = profiler.duplicate_values()

    assert result["duplicate_rows"] == 1
    assert result["duplicate_rate"] == 33.33


def test_completeness_score(sample_data):
    profiler = DataQualityProfiler(sample_data)

    assert profiler.completeness_score() == 93.75


def test_quality_score_for_perfect_dataset():
    data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "risk": ["low", "medium", "high"],
        }
    )

    profiler = DataQualityProfiler(data)

    assert profiler.completeness_score() == 100.0
    assert profiler.uniqueness_score() == 100.0
    assert profiler.quality_score() == 100.0


def test_invalid_input():
    with pytest.raises(TypeError):
        DataQualityProfiler([1, 2, 3])


def test_empty_dataframe():
    with pytest.raises(ValueError):
        DataQualityProfiler(pd.DataFrame())


def test_analyze_returns_expected_sections(sample_data):
    profiler = DataQualityProfiler(sample_data)
    report = profiler.analyze()

    assert "dataset" in report
    assert "missing_values" in report
    assert "duplicates" in report
    assert "data_types" in report
    assert "scores" in report
