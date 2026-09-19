"""
Core data quality profiler for TrustLens AI.

This module provides automated profiling of pandas DataFrames and
calculates foundational data-quality indicators used by TrustLens.
"""

from typing import Any, Dict

import pandas as pd


class DataQualityProfiler:
    """
    Analyse a pandas DataFrame and produce a data-quality report.

    The initial TrustLens quality model evaluates two foundational
    dimensions:

    - Completeness: the proportion of cells containing values.
    - Uniqueness: the proportion of rows that are not duplicates.

    These dimensions are combined into an initial Data Quality Score.
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
            If the supplied object is not a pandas DataFrame.
        ValueError
            If the supplied DataFrame contains no rows or columns.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty or data.shape[1] == 0:
            raise ValueError("data must contain at least one row and one column.")

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
        return self.row_count * self.column_count

    def missing_values(self) -> Dict[str, Any]:
        """Calculate dataset and column-level missing-value statistics."""
        missing_by_column = self.data.isna().sum()
        total_missing = int(missing_by_column.sum())
        missing_rate = (total_missing / self.total_cells) * 100

        column_report = {}

        for column in self.data.columns:
            count = int(missing_by_column[column])
            rate = (count / self.row_count) * 100

            column_report[str(column)] = {
                "missing_count": count,
                "missing_rate": round(rate, 2),
            }

        return {
            "total_missing": total_missing,
            "missing_rate": round(missing_rate, 2),
            "by_column": column_report,
        }

    def duplicate_values(self) -> Dict[str, float]:
        """Calculate duplicate-row statistics."""
        duplicate_count = int(self.data.duplicated().sum())
        duplicate_rate = (duplicate_count / self.row_count) * 100

        return {
            "duplicate_rows": duplicate_count,
            "duplicate_rate": round(duplicate_rate, 2),
        }

    def data_types(self) -> Dict[str, str]:
        """Return the detected pandas data type for each column."""
        return {
            str(column): str(dtype)
            for column, dtype in self.data.dtypes.items()
        }

    def completeness_score(self) -> float:
        """
        Calculate completeness on a 0-100 scale.

        A dataset containing no missing cells receives 100.
        """
        missing_rate = self.missing_values()["missing_rate"]
        return round(max(0.0, 100.0 - missing_rate), 2)

    def uniqueness_score(self) -> float:
        """
        Calculate row uniqueness on a 0-100 scale.

        A dataset containing no duplicated rows receives 100.
        """
        duplicate_rate = self.duplicate_values()["duplicate_rate"]
        return round(max(0.0, 100.0 - duplicate_rate), 2)

    def quality_score(self) -> float:
        """
        Calculate the initial TrustLens Data Quality Score.

        Version 0.1 uses equal weighting:

        50% completeness
        50% row uniqueness

        Additional dimensions will be introduced as the framework evolves.
        """
        score = (
            0.50 * self.completeness_score()
            + 0.50 * self.uniqueness_score()
        )

        return round(score, 2)

    def analyze(self) -> Dict[str, Any]:
        """Run the complete initial data-quality assessment."""
        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
                "total_cells": self.total_cells,
            },
            "missing_values": self.missing_values(),
            "duplicates": self.duplicate_values(),
            "data_types": self.data_types(),
            "scores": {
                "completeness": self.completeness_score(),
                "uniqueness": self.uniqueness_score(),
                "data_quality": self.quality_score(),
            },
        }
