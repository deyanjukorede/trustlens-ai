"""
Core data governance assessor for TrustLens AI.

This module provides the foundation for evaluating governance-related
characteristics of datasets used in artificial intelligence and
machine learning systems.

The assessor combines structural governance information with
sensitive-data detection, privacy-risk assessment, metadata
completeness assessment, and governance-controls assessment to
support broader privacy, metadata, governance-control, and
governance-scoring capabilities.
"""

from typing import Any, Dict, Optional

import pandas as pd

from .controls import GovernanceControlsAssessor
from .metadata import MetadataCompletenessAssessor
from .privacy_risk import PrivacyRiskAssessor
from .sensitive_data import SensitiveDataDetector


class DataGovernanceAssessor:
    """
    Perform governance assessment of a pandas DataFrame.

    The assessor provides structural information, sensitive-data
    analysis, privacy-risk assessment, optional metadata-completeness
    assessment, and optional governance-controls assessment.

    These components provide a foundation for broader TrustLens
    governance scoring and AI data-readiness assessment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        governance_controls: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise the governance assessor.

        Parameters
        ----------
        data:
            Dataset to be assessed.
        metadata:
            Optional dataset metadata used for metadata-completeness
            assessment.
        governance_controls:
            Optional mapping describing governance controls associated
            with the dataset.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame, metadata is invalid,
            or governance_controls is invalid.
        ValueError
            If the DataFrame is empty.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")

        if governance_controls is not None and not isinstance(
            governance_controls, dict
        ):
            raise TypeError(
                "governance_controls must be a dictionary or None"
            )

        self.data = data
        self.metadata = metadata
        self.governance_controls = governance_controls

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

    def metadata_completeness_analysis(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Assess completeness of governance metadata when metadata is supplied.

        Returns
        -------
        dict or None
            Structured metadata-completeness assessment when metadata
            is available, otherwise None.
        """
        if self.metadata is None:
            return None

        assessor = MetadataCompletenessAssessor(
            self.data,
            self.metadata,
        )
        return assessor.assess()

    def governance_controls_analysis(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Assess governance-control coverage when controls are supplied.

        Returns
        -------
        dict or None
            Structured governance-controls assessment when governance
            controls are available, otherwise None.
        """
        if self.governance_controls is None:
            return None

        assessor = GovernanceControlsAssessor(
            self.governance_controls
        )
        return assessor.assess()

    def assess(self) -> Dict[str, Any]:
        """
        Run the complete TrustLens governance assessment.

        Returns
        -------
        dict
            Structured governance assessment containing dataset
            information, column inventory, sensitive-data analysis,
            privacy-risk analysis, metadata-completeness analysis,
            and governance-controls analysis.
        """
        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "column_inventory": self.column_inventory(),
            "sensitive_data": self.sensitive_data_analysis(),
            "privacy_risk": self.privacy_risk_analysis(),
            "metadata_completeness": (
                self.metadata_completeness_analysis()
            ),
            "governance_controls": (
                self.governance_controls_analysis()
            ),
        }
