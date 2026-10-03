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
from trustlens.operational_trust.reproducibility import (
    ReproducibilityReadinessAnalyzer,
)
from trustlens.operational_trust.stability import DataStabilityAnalyzer


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
    high_missingness_threshold:
        Missing-value rate above which a feature requires stability
        review.
    near_constant_threshold:
        Dominant-value proportion above which a non-constant feature
        is considered near-constant.
    high_cardinality_threshold:
        Unique-value proportion above which a categorical feature may
        require stability review.
    duplicate_rate_threshold:
        Duplicate-row rate above which duplicate pressure requires
        stability review.
    reproducibility_metadata:
        Optional dictionary containing evidence that can support
        reproducibility assessment, such as dataset version, code
        version, random seed, environment, source, lineage, and
        execution identifier.
    identifier_column:
        Optional dataset column intended to provide row-level
        identification for reproducibility analysis.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        numeric_mean_shift_threshold: float = 0.20,
        categorical_distribution_threshold: float = 0.20,
        high_missingness_threshold: float = 0.50,
        near_constant_threshold: float = 0.95,
        high_cardinality_threshold: float = 0.90,
        duplicate_rate_threshold: float = 0.10,
        reproducibility_metadata: Optional[Dict[str, Any]] = None,
        identifier_column: Optional[str] = None,
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

        if (
            reproducibility_metadata is not None
            and not isinstance(reproducibility_metadata, dict)
        ):
            raise TypeError(
                "reproducibility_metadata must be a dictionary "
                "when supplied"
            )

        if identifier_column is not None:
            if not isinstance(identifier_column, str):
                raise TypeError(
                    "identifier_column must be a string"
                )

            if not identifier_column.strip():
                raise ValueError(
                    "identifier_column must not be empty"
                )

            if identifier_column not in data.columns:
                raise ValueError(
                    "identifier_column must exist in data"
                )

        self._validate_threshold(
            "numeric_mean_shift_threshold",
            numeric_mean_shift_threshold,
        )
        self._validate_threshold(
            "categorical_distribution_threshold",
            categorical_distribution_threshold,
        )
        self._validate_threshold(
            "high_missingness_threshold",
            high_missingness_threshold,
        )
        self._validate_threshold(
            "near_constant_threshold",
            near_constant_threshold,
        )
        self._validate_threshold(
            "high_cardinality_threshold",
            high_cardinality_threshold,
        )
        self._validate_threshold(
            "duplicate_rate_threshold",
            duplicate_rate_threshold,
        )

        self.data = data.copy()
        self.reference_data = (
            reference_data.copy()
            if reference_data is not None
            else None
        )

        self.reproducibility_metadata = (
            dict(reproducibility_metadata)
            if reproducibility_metadata is not None
            else {}
        )
        self.identifier_column = identifier_column

        self.numeric_mean_shift_threshold = float(
            numeric_mean_shift_threshold
        )
        self.categorical_distribution_threshold = float(
            categorical_distribution_threshold
        )
        self.high_missingness_threshold = float(
            high_missingness_threshold
        )
        self.near_constant_threshold = float(
            near_constant_threshold
        )
        self.high_cardinality_threshold = float(
            high_cardinality_threshold
        )
        self.duplicate_rate_threshold = float(
            duplicate_rate_threshold
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

    @staticmethod
    def _validate_threshold(
        name: str,
        value: float,
    ) -> None:
        """Validate a proportion-based Operational Trust threshold."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")

        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} must be between 0 and 1"
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
            "reproducibility_metadata_supplied": bool(
                self.reproducibility_metadata
            ),
            "identifier_column": self.identifier_column,
        }

    def analysis_availability(self) -> Dict[str, Dict[str, Any]]:
        """Report which Operational Trust analyses have required context."""
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
                "optional_context": [
                    "reproducibility_metadata",
                    "identifier_column",
                ],
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

    def _assess_data_stability(self) -> Dict[str, Any]:
        """Run data stability analysis on the current dataset."""
        analyzer = DataStabilityAnalyzer(
            data=self.data,
            high_missingness_threshold=(
                self.high_missingness_threshold
            ),
            near_constant_threshold=(
                self.near_constant_threshold
            ),
            high_cardinality_threshold=(
                self.high_cardinality_threshold
            ),
            duplicate_rate_threshold=(
                self.duplicate_rate_threshold
            ),
        )

        return analyzer.assess()

    def _assess_reproducibility(self) -> Dict[str, Any]:
        """Run reproducibility and reliability readiness analysis."""
        analyzer = ReproducibilityReadinessAnalyzer(
            data=self.data,
            metadata=self.reproducibility_metadata,
            identifier_column=self.identifier_column,
        )

        return analyzer.assess()

    def assess(self) -> Dict[str, Any]:
        """Return the integrated Operational Trust assessment."""
        data_drift = self._assess_data_drift()
        data_stability = self._assess_data_stability()
        reproducibility = self._assess_reproducibility()

        review_reasons = []

        if (
            data_drift is not None
            and data_drift["review_required"]
        ):
            review_reasons.append("data_drift")

        if data_stability["review_required"]:
            review_reasons.append("data_stability")

        if reproducibility["review_required"]:
            review_reasons.append("reproducibility")

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
            "data_stability": data_stability,
            "reproducibility": reproducibility,
            "operational_trust_summary": {
                "status": status,
                "review_required": review_required,
                "review_reasons": review_reasons,
            },
        }
