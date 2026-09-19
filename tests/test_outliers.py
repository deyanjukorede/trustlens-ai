import pandas as pd
import pytest

from trustlens.quality.outliers import OutlierDetector


def test_detects_obvious_outlier():
    """An extreme value should be detected using the IQR method."""
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

    detector = OutlierDetector(data)
    result = detector.detect_iqr()

    assert result["transaction_amount"]["outlier_count"] == 1
    assert result["transaction_amount"]["outlier_values"] == [50000]
    assert result["transaction_amount"]["outlier_indices"] == [9]


def test_dataset_without_outliers():
    """A stable numerical dataset should contain no IQR outliers."""
    data = pd.DataFrame(
        {
            "score": [10, 11, 12, 13, 14, 15, 16]
        }
    )

    detector = OutlierDetector(data)
    result = detector.detect_iqr()

    assert result["score"]["outlier_count"] == 0
    assert result["score"]["outlier_values"] == []
    assert detector.affected_columns() == []


def test_missing_values_are_ignored():
    """Missing values should not prevent outlier analysis."""
    data = pd.DataFrame(
        {
            "amount": [
                100,
                101,
                None,
                99,
                102,
                98,
                100,
                50000,
            ]
        }
    )

    detector = OutlierDetector(data)
    result = detector.detect_iqr()

    assert result["amount"]["outlier_count"] == 1
    assert result["amount"]["outlier_values"] == [50000.0]


def test_non_numeric_columns_are_ignored():
    """Outlier detection should only analyse numerical columns."""
    data = pd.DataFrame(
        {
            "customer": ["A", "B", "C", "D"],
            "segment": ["retail", "retail", "business", "retail"],
            "amount": [100, 101, 99, 102],
        }
    )

    detector = OutlierDetector(data)

    assert detector.numeric_columns == ["amount"]


def test_invalid_input():
    """Non-DataFrame input should be rejected."""
    with pytest.raises(TypeError):
        OutlierDetector([1, 2, 3])


def test_empty_dataframe():
    """An empty DataFrame should be rejected."""
    with pytest.raises(ValueError):
        OutlierDetector(pd.DataFrame())


def test_invalid_multiplier():
    """The IQR multiplier must be greater than zero."""
    data = pd.DataFrame({"amount": [10, 20, 30]})
    detector = OutlierDetector(data)

    with pytest.raises(ValueError):
        detector.detect_iqr(multiplier=0)


def test_summary():
    """Summary should report the overall outlier assessment."""
    data = pd.DataFrame(
        {
            "amount": [
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
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )

    detector = OutlierDetector(data)
    summary = detector.summary()

    assert summary["method"] == "IQR"
    assert summary["numeric_columns_analyzed"] == 2
    assert summary["columns_with_outliers"] == 1
    assert summary["affected_columns"] == ["amount"]
    assert summary["total_outliers_detected"] == 1
    assert "details" in summary
