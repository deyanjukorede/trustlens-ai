"""
Integrated Operational Trust assessment for TrustLens AI.

This module provides the central orchestration layer for Operational
Trust analysis. Individual analytical components such as data drift,
data stability, reproducibility, and monitoring readiness are integrated
here as they become available.

Operational Trust indicators are intended to support investigation and
human review. They do not independently establish that a data or AI
system is reliable, trustworthy, production-ready, or compliant.
"""

from typing import Any, Dict, Optional

import pandas as pd

from trustlens.operational_trust.drift import DataDriftAnalyzer


class OperationalTrustAssessor:
    """
    Coordinate Operational Trust analysis for a dataset.

    Parameters
    ----------
    data:
        Current dataset being assessed.
    reference_data:
        Optional reference dataset representing an earlier, baseline,
        training, or otherwise relevant comparison population.
    numeric_mean_shift_threshold:
        Relative mean-shift threshold used by numeric drift analysis.
    categorical_distribution_threshold:
        Maximum absolute category-proportion change used by categorical
        drift analysis.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        numeric_mean_shift_threshold: float = 0.20,
        categorical_distribution_threshold: float = 0.20,
    ) -> None:
        """Initialise the Operational Trust assessor."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if reference_data is not None:
            if not isinstance(reference_data, pd.DataFrame):
                raise TypeError(
                    "reference_data must be a pandas DataFrame"
                )

            if reference_data.empty:
                raise ValueError("reference_data must not be empty")

        if not 0 <= numeric_mean_shift_threshold <= 1:
            raise ValueError(
                "numeric_mean_shift_threshold must be between 0 and 1"
            )

        if not 0 <= categorical_distribution_threshold <= 1:
            raise ValueError(
                "categorical_distribution_threshold must be between 0 and 1"
            )

        self.data = data.copy()
        self.reference_data = (
            reference_data.copy()
            if reference_data is not None
            else None
        )

        self.numeric_mean_shift_threshold = float(
            numeric_mean_shift_threshold
        )

        self.categorical_distribution_threshold = float(
            categorical_distribution_threshold
        )

        self.row_count = int(len(self.data))
        self.column_count = int(len(self.data.columns))

        self.reference_row_count = (
            int(len(self.reference_data))
            if self.reference_data is not None
            else 0
        )

        self.reference_column_count = (
            int(len(self.reference_data.columns))
            if self.reference_data is not None
            else 0
        )

    @property
    def reference_data_available(self) -> bool:
        """Return whether a reference dataset is available."""
        return self.reference_data is not None

    def assessment_context(self) -> Dict[str, Any]:
        """Describe the dataset context available for analysis."""
        return {
            "current_data": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "reference_data_available": self.reference_data_available,
            "reference_data": (
                {
                    "rows": self.reference_row_count,
                    "columns": self.reference_column_count,
                }
                if self.reference_data_available
                else None
            ),
        }

    def analysis_availability(self) -> Dict[str, Dict[str, Any]]:
        """
        Report which Operational Trust analyses have required context.

        More detailed availability rules will be introduced as the
        specialised Operational Trust analyzers are implemented.
        """
        return {
            "data_drift": {
                "available": self.reference_data_available,
                "requires": ["current_data", "reference_data"],
            },
            "data_stability": {
                "available": True,
                "requires": ["current_data"],
            },
            "reproducibility": {
                "available": True,
                "requires": ["current_data"],
            },
            "monitoring_readiness": {
                "available": True,
                "requires": ["current_data"],
            },
        }

    def _assess_data_drift(self) -> Optional[Dict[str, Any]]:
        """Run data drift analysis when reference data is available."""
        if not self.reference_data_available:
            return None

        analyzer = DataDriftAnalyzer(
            reference_data=self.reference_data,
            current_data=self.data,
            numeric_mean_shift_threshold=(
                self.numeric_mean_shift_threshold
            ),
            categorical_distribution_threshold=(
                self.categorical_distribution_threshold
            ),
        )

        return analyzer.assess()

    def assess(self) -> Dict[str, Any]:
        """
        Return the integrated Operational Trust assessment.

        Data drift is assessed when a reference dataset is available.
        Other Operational Trust components are added as Phase 6
        progresses.
        """
        data_drift = self._assess_data_drift()

        review_reasons = []

        if (
            data_drift is not None
            and data_drift["review_required"]
        ):
            review_reasons.append("data_drift")

        review_required = bool(review_reasons)

        status = (
            "review"
            if review_required
            else "no_review_indicators"
        )

        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "reference_dataset": (
                {
                    "rows": self.reference_row_count,
                    "columns": self.reference_column_count,
                }
                if self.reference_data_available
                else None
            ),
            "assessment_context": self.assessment_context(),
            "analysis_availability": self.analysis_availability(),
            "data_drift": data_drift,
            "operational_trust_summary": {
                "status": status,
                "review_required": review_required,
                "review_reasons": review_reasons,
            },
        }
