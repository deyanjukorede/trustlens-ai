"""
Core data governance assessor for TrustLens AI.

This module provides the foundation for evaluating governance-related
characteristics of datasets used in artificial intelligence and
machine learning systems.

More advanced governance capabilities such as sensitive-data
detection, privacy-risk assessment, metadata validation, and
governance scoring will be added in later TrustLens phases.
"""

from typing import Any, Dict

import pandas as pd


class DataGovernanceAssessor:
    """
    Perform foundational governance assessment of a pandas DataFrame.

    The assessor provides structural information that later TrustLens
    governance components can use when evaluating data classification,
    sensitive information, metadata, privacy risk, and governance
    controls.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialise the governance assessor.

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

    def column_inventory(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a basic governance inventory of dataset columns.

        Returns
        -------
        dict
            Mapping of column names to basic structural metadata.
        """
        inventory: Dict[str, Dict[str, Any]] = {}

        for column in self.data.columns:
            series = self.data[column]

            inventory[str(column)] = {
                "data_type": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            }

        return inventory

    def assess(self) -> Dict[str, Any]:
        """
        Run the foundational TrustLens governance assessment.

        Returns
        -------
        dict
            Structured governance assessment containing dataset
            information and a column inventory.
        """
        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "column_inventory": self.column_inventory(),
        }
