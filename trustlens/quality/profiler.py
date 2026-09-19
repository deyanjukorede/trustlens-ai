"""
Core data quality profiler for TrustLens AI.

This module provides automated profiling of pandas DataFrames and
calculates foundational data-quality indicators used by TrustLens.
"""

from typing import Any, Dict

import pandas as pd

from .outliers import OutlierDetector


class DataQualityProfiler:
    """
    Analyse a pandas DataFrame and produce a data-quality report.

    The TrustLens quality model currently evaluates:

    - Dataset structure
    - Missing values
    - Duplicate rows
    - Data types
    - Completeness
    - Row uniqueness
    - Numerical outliers

    These indicators provide the foundation for the broader
    TrustLens AI Data Readiness Score.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialise the profiler with a pandas DataFrame.

        Parameters
        ----------
        data:
            Dataset to be assessed.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame.
        ValueError
            If the DataFrame is empty.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        self.data = data

    @property
    def row_count(self) -> int:
        """Return the number of rows in the dataset."""
        return int(self.data.shape[0])

    @property
    def column_count(self) -> int:
        """Return the number of columns in the dataset."""
        return int(self.data.shape[1])

    @property
    def total_cells(self) -> int:
        """Return the total number of cells in the dataset."""
        return int(self.data.size)

    def missing_values(self) -> Dict[str, Any]:
        """
        Analyse missing values in the dataset.

        Returns
        -------
        dict
            Missing-value counts and percentages.
        """
        missing_by_column = self.data.isna().sum()
        total_missing = int(missing_by_column.sum())

        if self.total_cells == 0:
            missing_rate = 0.0
        else:
            missing_rate = round(
                (total_missing / self.total_cells) * 100,
                2,
            )

        return {
            "total_missing": total_missing,
            "missing_rate": missing_rate,
            "by_column": {
                column: int(count)
                for column, count in missing_by_column.items()
            },
        }

    def duplicate_values(self) -> Dict[str, Any]:
        """
        Analyse duplicated rows in the dataset.

        Returns
        -------
        dict
            Duplicate-row count and percentage.
        """
        duplicate_count = int(self.data.duplicated().sum())

        if self.row_count == 0:
            duplicate_rate = 0.0
        else:
            duplicate_rate = round(
                (duplicate_count / self.row_count) * 100,
                2,
            )

        return {
            "duplicate_count": duplicate_count,
            "duplicate_rate": duplicate_rate,
        }

    def data_types(self) -> Dict[str, str]:
        """
        Return pandas data types for all columns.

        Returns
        -------
        dict
            Mapping of column names to data types.
        """
        return {
            column: str(dtype)
            for column, dtype in self.data.dtypes.items()
        }

    def completeness_score(self) -> float:
        """
        Calculate completeness on a 0-100 scale.

        A dataset containing no missing values receives 100.
        """
        missing_rate = self.missing_values()["missing_rate"]

        return round(
            max(0.0, 100.0 - missing_rate),
            2,
        )

    def uniqueness_score(self) -> float:
        """
        Calculate row uniqueness on a 0-100 scale.

        A dataset containing no duplicated rows receives 100.
        """
        duplicate_rate = self.duplicate_values()["duplicate_rate"]

        return round(
            max(0.0, 100.0 - duplicate_rate),
            2,
        )

    def outlier_analysis(self) -> Dict[str, Any]:
        """
        Analyse numerical columns for potential outliers.

        Uses the TrustLens IQR-based OutlierDetector and returns
        a high-level summary suitable for inclusion in the
        data-quality assessment report.

        Returns
        -------
        dict
            Summary of numerical outliers detected in the dataset.
        """
        detector = OutlierDetector(self.data)

        return detector.summary()

    def quality_score(self) -> float:
        """
        Calculate the TrustLens Data Quality Score.

        Version 0.1 uses equal weighting:

        - 50% completeness
        - 50% row uniqueness

        Outlier analysis is currently reported separately rather than
        included directly in the numerical score. This keeps the scoring
        methodology transparent while the framework evolves.
        """
        score = (
            0.50 * self.completeness_score()
            + 0.50 * self.uniqueness_score()
        )

        return round(score, 2)

    def analyze(self) -> Dict[str, Any]:
        """
        Run the complete TrustLens data-quality assessment.

        Returns
        -------
        dict
            Structured data-quality report containing dataset statistics,
            missing values, duplicates, data types, outlier analysis,
            and quality scores.
        """
        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
                "total_cells": self.total_cells,
            },
            "missing_values": self.missing_values(),
            "duplicates": self.duplicate_values(),
            "data_types": self.data_types(),
            "outliers": self.outlier_analysis(),
            "scores": {
                "completeness": self.completeness_score(),
                "uniqueness": self.uniqueness_score(),
                "data_quality": self.quality_score(),
            },
        }
