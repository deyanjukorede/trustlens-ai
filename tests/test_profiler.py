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
    """Profiler should correctly identify dataset dimensions."""
    profiler = DataQualityProfiler(sample_data)

    assert profiler.row_count == 4
    assert profiler.column_count == 4
    assert profiler.total_cells == 16


def test_missing_values(sample_data):
    """Profiler should correctly identify missing values."""
    profiler = DataQualityProfiler(sample_data)
    result = profiler.missing_values()

    assert result["total_missing"] == 1
    assert result["missing_rate"] == 6.25
    assert result["by_column"]["age"]["missing_count"] == 1
    assert result["by_column"]["age"]["missing_rate"] == 25.0


def test_duplicate_rows():
    """Profiler should correctly identify duplicate rows."""
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
    """Completeness score should reflect missing data."""
    profiler = DataQualityProfiler(sample_data)

    assert profiler.completeness_score() == 93.75


def test_quality_score_for_perfect_dataset():
    """A complete and unique dataset should receive a perfect quality score."""
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
    """Profiler should reject non-DataFrame input."""
    with pytest.raises(TypeError):
        DataQualityProfiler([1, 2, 3])


def test_empty_dataframe():
    """Profiler should reject an empty DataFrame."""
    with pytest.raises(ValueError):
        DataQualityProfiler(pd.DataFrame())


def test_outlier_analysis_detects_obvious_outlier():
    """Profiler should expose outlier information from OutlierDetector."""
    data = pd.DataFrame(
        {
            "transaction_amount": [
                100,
                102,
                98,
                101,
                99,
                103,
                100,
                102,
                101,
                50000,
            ],
            "category": [
                "A",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
        }
    )

    profiler = DataQualityProfiler(data)
    result = profiler.outlier_analysis()

    assert result["method"] == "IQR"
    assert result["numeric_columns_analyzed"] == 1
    assert result["columns_with_outliers"] == 1
    assert "transaction_amount" in result["affected_columns"]
    assert result["total_outliers_detected"] == 1


def test_analyze_includes_outlier_section():
    """Full profiler report should include the outlier assessment."""
    data = pd.DataFrame(
        {
            "transaction_amount": [
                100,
                102,
                98,
                101,
                99,
                103,
                100,
                102,
                101,
                50000,
            ],
            "category": [
                "A",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
        }
    )

    profiler = DataQualityProfiler(data)
    report = profiler.analyze()

    assert "outliers" in report
    assert report["outliers"]["method"] == "IQR"
    assert report["outliers"]["total_outliers_detected"] == 1
    assert "transaction_amount" in report["outliers"]["affected_columns"]


def test_outlier_analysis_with_no_outliers():
    """Profiler should report zero outliers for stable numerical data."""
    data = pd.DataFrame(
        {
            "score": [10, 11, 12, 13, 14, 15, 16],
            "label": ["A", "B", "C", "D", "E", "F", "G"],
        }
    )

    profiler = DataQualityProfiler(data)
    result = profiler.outlier_analysis()

    assert result["method"] == "IQR"
    assert result["numeric_columns_analyzed"] == 1
    assert result["columns_with_outliers"] == 0
    assert result["affected_columns"] == []
    assert result["total_outliers_detected"] == 0


def test_quality_score_remains_separate_from_outliers():
    """
    Outlier reporting should not alter the current quality-score formula.

    The current TrustLens quality score is based on completeness and
    uniqueness, while outliers are reported as a separate assessment.
    """
    data = pd.DataFrame(
        {
            "transaction_amount": [
                100,
                102,
                98,
                101,
                99,
                103,
                100,
                102,
                101,
                50000,
            ]
        }
    )

    profiler = DataQualityProfiler(data)

    assert profiler.completeness_score() == 100.0
    assert profiler.uniqueness_score() == 100.0
    assert profiler.quality_score() == 100.0
    assert profiler.outlier_analysis()["total_outliers_detected"] == 1


def test_analyze_returns_expected_sections(sample_data):
    """Full analysis should return every expected profiler section."""
    profiler = DataQualityProfiler(sample_data)
    report = profiler.analyze()

    assert "dataset" in report
    assert "missing_values" in report
    assert "duplicates" in report
    assert "data_types" in report
    assert "outliers" in report
    assert "scores" in report
