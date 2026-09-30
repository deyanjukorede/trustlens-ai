"""
Core data governance assessor for TrustLens AI.

This module provides the foundation for evaluating governance-related
characteristics of datasets used in artificial intelligence and
machine learning systems.

The assessor combines structural governance information with
sensitive-data detection and privacy-risk assessment to support
broader privacy, metadata, governance-control, and governance-scoring
capabilities.
"""

from typing import Any, Dict

import pandas as pd

from .privacy_risk import PrivacyRiskAssessor
from .sensitive_data import SensitiveDataDetector


class DataGovernanceAssessor:
    """
    Perform governance assessment of a pandas DataFrame.

    The assessor provides structural information, sensitive-data
    analysis, and privacy-risk assessment that later TrustLens
    governance components can use when evaluating metadata quality,
    governance controls, governance scoring, and overall governance
    readiness.
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

    def sensitive_data_analysis(self) -> Dict[str, Any]:
        """
        Analyse the dataset for potentially sensitive information.

        Detection combines recognised column-name indicators with
        value-pattern analysis.

        Returns
        -------
        dict
            Sensitive-data assessment summary.
        """
        detector = SensitiveDataDetector(self.data)
        return detector.summary()

    def privacy_risk_analysis(self) -> Dict[str, Any]:
        """
        Assess privacy risk associated with the dataset.

        The privacy-risk assessor builds on sensitive-data detection
        to calculate privacy-risk indicators, risk classification,
        exposure information, and governance recommendations.

        Returns
        -------
        dict
            Structured privacy-risk assessment.
        """
        assessor = PrivacyRiskAssessor(self.data)
        return assessor.assess()

    def assess(self) -> Dict[str, Any]:
        """
        Run the complete TrustLens governance assessment.

        Returns
        -------
        dict
            Structured governance assessment containing dataset
            information, column inventory, sensitive-data analysis,
            and privacy-risk analysis.
        """
        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "column_inventory": self.column_inventory(),
            "sensitive_data": self.sensitive_data_analysis(),
            "privacy_risk": self.privacy_risk_analysis(),
        }
