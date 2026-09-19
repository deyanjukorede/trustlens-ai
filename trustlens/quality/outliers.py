"""
Outlier detection utilities for TrustLens AI.

This module identifies unusual numerical observations using
statistical methods that do not require a machine-learning model.
"""

from typing import Any, Dict, List

import pandas as pd


class OutlierDetector:
    """
    Detect numerical outliers in a pandas DataFrame.

    The initial TrustLens implementation uses the Interquartile Range
    (IQR) method because it is simple, interpretable, and robust to
    non-normal distributions.
    """

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")

        if dataframe.empty:
            raise ValueError("dataframe must not be empty")

        self.dataframe = dataframe
        self.numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

    def detect_iqr(
        self,
        multiplier: float = 1.5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using the Interquartile Range (IQR) method.

        Values below Q1 - multiplier * IQR or above
        Q3 + multiplier * IQR are classified as outliers.
        """

        if multiplier <= 0:
            raise ValueError("multiplier must be greater than zero")

        results: Dict[str, Dict[str, Any]] = {}

        for column in self.numeric_columns:
            series = self.dataframe[column].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            mask = (series < lower_bound) | (series > upper_bound)
            outliers = series[mask]

            results[column] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": int(outliers.shape[0]),
                "outlier_rate": round(
                    (outliers.shape[0] / series.shape[0]) * 100,
                    2,
                ),
                "outlier_indices": outliers.index.tolist(),
                "outlier_values": outliers.tolist(),
            }

        return results

    def affected_columns(self) -> List[str]:
        """Return numerical columns containing at least one outlier."""

        analysis = self.detect_iqr()

        return [
            column
            for column, result in analysis.items()
            if result["outlier_count"] > 0
        ]

    def summary(self) -> Dict[str, Any]:
        """Return a high-level outlier assessment."""

        analysis = self.detect_iqr()

        total_outliers = sum(
            result["outlier_count"]
            for result in analysis.values()
        )

        affected = [
            column
            for column, result in analysis.items()
            if result["outlier_count"] > 0
        ]

        return {
            "method": "IQR",
            "numeric_columns_analyzed": len(self.numeric_columns),
            "columns_with_outliers": len(affected),
            "affected_columns": affected,
            "total_outliers_detected": total_outliers,
            "details": analysis,
        }
