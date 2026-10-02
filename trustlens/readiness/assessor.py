"""
Core AI readiness assessor for TrustLens AI.

This module provides the foundation for evaluating whether datasets
are structurally suitable for artificial intelligence and machine
learning workflows.

More advanced Phase 4 capabilities, including class-imbalance
analysis, feature suitability, leakage-risk detection, and integrated
AI readiness scoring, are added through the readiness package.
"""

from typing import Any, Dict, List

import pandas as pd


class AIReadinessAssessor:
    """
    Perform foundational AI readiness assessment of a pandas DataFrame.

    The assessor establishes structural information used by later
    TrustLens readiness components when evaluating modelling
    suitability, target variables, feature composition, imbalance,
    and potential leakage risks.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> None:
        """
        Initialise the AI readiness assessor.

        Parameters
        ----------
        data:
            Dataset to be assessed.
        target_column:
            Optional name of the intended prediction target.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame.
        ValueError
            If the DataFrame is empty or the supplied target column
            does not exist.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if target_column is not None and target_column not in data.columns:
            raise ValueError(
                f"target_column '{target_column}' does not exist in the dataset"
            )

        self.data = data
        self.target_column = target_column

    @property
    def row_count(self) -> int:
        """Return the number of rows in the dataset."""
        return int(self.data.shape[0])

    @property
    def column_count(self) -> int:
        """Return the number of columns in the dataset."""
        return int(self.data.shape[1])

    @property
    def feature_columns(self) -> List[str]:
        """
        Return columns currently treated as modelling features.

        When a target column is supplied, it is excluded from the
        feature list.
        """
        columns = [str(column) for column in self.data.columns]

        if self.target_column is None:
            return columns

        return [
            column
            for column in columns
            if column != self.target_column
        ]

    def feature_type_summary(self) -> Dict[str, List[str]]:
        """
        Group modelling features into broad structural data types.

        Returns
        -------
        dict
            Numeric, categorical, boolean, and datetime feature lists.
        """
        summary: Dict[str, List[str]] = {
            "numeric": [],
            "categorical": [],
            "boolean": [],
            "datetime": [],
        }

        for column in self.feature_columns:
            series = self.data[column]

            if pd.api.types.is_bool_dtype(series):
                summary["boolean"].append(column)
            elif pd.api.types.is_numeric_dtype(series):
                summary["numeric"].append(column)
            elif pd.api.types.is_datetime64_any_dtype(series):
                summary["datetime"].append(column)
            else:
                summary["categorical"].append(column)

        return summary

    def missing_value_summary(self) -> Dict[str, Any]:
        """
        Summarise missing values relevant to modelling readiness.
        """
        missing_by_column = {
            str(column): int(count)
            for column, count in self.data.isna().sum().items()
        }

        columns_with_missing = [
            column
            for column, count in missing_by_column.items()
            if count > 0
        ]

        total_cells = self.row_count * self.column_count
        total_missing = sum(missing_by_column.values())

        missing_rate = (
            (total_missing / total_cells) * 100.0
            if total_cells
            else 0.0
        )

        return {
            "total_missing_values": int(total_missing),
            "missing_rate": round(float(missing_rate), 2),
            "columns_with_missing_values": columns_with_missing,
            "missing_by_column": missing_by_column,
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the foundational TrustLens AI readiness assessment.

        Returns
        -------
        dict
            Structured readiness information that later Phase 4
            components can extend.
        """
        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "target_column": self.target_column,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "feature_types": self.feature_type_summary(),
            "missing_values": self.missing_value_summary(),
        }
