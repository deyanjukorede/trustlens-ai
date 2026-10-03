"""
Data drift analysis for TrustLens AI.

This module compares a current dataset with a reference dataset and
reports measurable changes in numeric and categorical feature
distributions.

Drift indicators are intended to support monitoring and human review.
They do not independently establish that a model or data system is
unreliable, unsafe, or unsuitable for continued use.
"""

from typing import Any, Dict, List

import pandas as pd


class DataDriftAnalyzer:
    """
    Analyse feature-level changes between reference and current data.

    Parameters
    ----------
    reference_data:
        Baseline dataset used for comparison.
    current_data:
        Current dataset being assessed.
    numeric_mean_shift_threshold:
        Relative mean-shift threshold used to flag numeric features.
    categorical_distribution_threshold:
        Maximum absolute category-proportion change used to flag
        categorical features.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        numeric_mean_shift_threshold: float = 0.20,
        categorical_distribution_threshold: float = 0.20,
    ) -> None:
        """Initialise the data drift analyzer."""
        if not isinstance(reference_data, pd.DataFrame):
            raise TypeError(
                "reference_data must be a pandas DataFrame"
            )

        if not isinstance(current_data, pd.DataFrame):
            raise TypeError(
                "current_data must be a pandas DataFrame"
            )

        if reference_data.empty:
            raise ValueError("reference_data must not be empty")

        if current_data.empty:
            raise ValueError("current_data must not be empty")

        if not 0 <= numeric_mean_shift_threshold <= 1:
            raise ValueError(
                "numeric_mean_shift_threshold must be between 0 and 1"
            )

        if not 0 <= categorical_distribution_threshold <= 1:
            raise ValueError(
                "categorical_distribution_threshold must be between 0 and 1"
            )

        self.reference_data = reference_data.copy()
        self.current_data = current_data.copy()

        self.numeric_mean_shift_threshold = float(
            numeric_mean_shift_threshold
        )

        self.categorical_distribution_threshold = float(
            categorical_distribution_threshold
        )

    def column_alignment(self) -> Dict[str, List[str]]:
        """Report shared and dataset-specific columns."""
        reference_columns = list(self.reference_data.columns)
        current_columns = list(self.current_data.columns)

        shared_columns = [
            column
            for column in reference_columns
            if column in self.current_data.columns
        ]

        reference_only = [
            column
            for column in reference_columns
            if column not in self.current_data.columns
        ]

        current_only = [
            column
            for column in current_columns
            if column not in self.reference_data.columns
        ]

        return {
            "shared_columns": shared_columns,
            "reference_only_columns": reference_only,
            "current_only_columns": current_only,
        }

    def _feature_type(self, column: str) -> str:
        """Determine the supported comparison type for a shared feature."""
        reference_series = self.reference_data[column]
        current_series = self.current_data[column]

        reference_numeric = pd.api.types.is_numeric_dtype(
            reference_series
        )

        current_numeric = pd.api.types.is_numeric_dtype(
            current_series
        )

        if reference_numeric and current_numeric:
            return "numeric"

        reference_datetime = pd.api.types.is_datetime64_any_dtype(
            reference_series
        )

        current_datetime = pd.api.types.is_datetime64_any_dtype(
            current_series
        )

        if reference_datetime and current_datetime:
            return "datetime"

        if reference_numeric != current_numeric:
            return "incompatible"

        if reference_datetime != current_datetime:
            return "incompatible"

        return "categorical"

    def _numeric_analysis(self, column: str) -> Dict[str, Any]:
        """Analyse mean movement for a numeric feature."""
        reference_series = self.reference_data[column].dropna()
        current_series = self.current_data[column].dropna()

        reference_mean = (
            float(reference_series.mean())
            if not reference_series.empty
            else 0.0
        )

        current_mean = (
            float(current_series.mean())
            if not current_series.empty
            else 0.0
        )

        absolute_mean_difference = abs(
            current_mean - reference_mean
        )

        if reference_mean == 0:
            relative_mean_shift = (
                0.0
                if current_mean == 0
                else 1.0
            )
        else:
            relative_mean_shift = (
                absolute_mean_difference
                / abs(reference_mean)
            )

        relative_mean_shift = round(
            float(relative_mean_shift),
            4,
        )

        requires_review = (
            relative_mean_shift
            > self.numeric_mean_shift_threshold
        )

        return {
            "reference_mean": round(reference_mean, 4),
            "current_mean": round(current_mean, 4),
            "absolute_mean_difference": round(
                float(absolute_mean_difference),
                4,
            ),
            "relative_mean_shift": relative_mean_shift,
            "threshold": self.numeric_mean_shift_threshold,
            "requires_review": requires_review,
        }

    def _categorical_analysis(
        self,
        column: str,
    ) -> Dict[str, Any]:
        """Analyse category-distribution movement for a feature."""
        reference_series = self.reference_data[column].dropna()
        current_series = self.current_data[column].dropna()

        reference_distribution = (
            reference_series.value_counts(normalize=True)
        )

        current_distribution = (
            current_series.value_counts(normalize=True)
        )

        categories = list(
            dict.fromkeys(
                list(reference_distribution.index)
                + list(current_distribution.index)
            )
        )

        category_changes: Dict[Any, Dict[str, float]] = {}

        maximum_distribution_change = 0.0

        for category in categories:
            reference_rate = float(
                reference_distribution.get(category, 0.0)
            )

            current_rate = float(
                current_distribution.get(category, 0.0)
            )

            absolute_change = abs(
                current_rate - reference_rate
            )

            maximum_distribution_change = max(
                maximum_distribution_change,
                absolute_change,
            )

            category_changes[category] = {
                "reference_rate": round(reference_rate, 4),
                "current_rate": round(current_rate, 4),
                "absolute_change": round(
                    float(absolute_change),
                    4,
                ),
            }

        maximum_distribution_change = round(
            float(maximum_distribution_change),
            4,
        )

        requires_review = (
            maximum_distribution_change
            > self.categorical_distribution_threshold
        )

        return {
            "category_changes": category_changes,
            "maximum_distribution_change": (
                maximum_distribution_change
            ),
            "threshold": (
                self.categorical_distribution_threshold
            ),
            "requires_review": requires_review,
        }

    def missing_data_summary(
        self,
        column: str,
    ) -> Dict[str, Any]:
        """Compare missingness for a shared feature."""
        reference_missing_count = int(
            self.reference_data[column].isna().sum()
        )

        current_missing_count = int(
            self.current_data[column].isna().sum()
        )

        reference_missing_rate = round(
            reference_missing_count
            / len(self.reference_data),
            4,
        )

        current_missing_rate = round(
            current_missing_count
            / len(self.current_data),
            4,
        )

        return {
            "reference_missing_count": reference_missing_count,
            "current_missing_count": current_missing_count,
            "reference_missing_rate": reference_missing_rate,
            "current_missing_rate": current_missing_rate,
            "missing_rate_change": round(
                abs(
                    current_missing_rate
                    - reference_missing_rate
                ),
                4,
            ),
        }

    def assess(self) -> Dict[str, Any]:
        """Return the complete feature-level drift assessment."""
        alignment = self.column_alignment()

        feature_analysis: Dict[str, Dict[str, Any]] = {}
        features_requiring_review: List[str] = []
        incompatible_features: List[str] = []

        for column in alignment["shared_columns"]:
            feature_type = self._feature_type(column)

            feature_result: Dict[str, Any] = {
                "feature_type": feature_type,
                "missing_data": self.missing_data_summary(
                    column
                ),
            }

            if feature_type == "numeric":
                analysis = self._numeric_analysis(column)
                feature_result["drift_analysis"] = analysis

                if analysis["requires_review"]:
                    features_requiring_review.append(column)

            elif feature_type == "categorical":
                analysis = self._categorical_analysis(column)
                feature_result["drift_analysis"] = analysis

                if analysis["requires_review"]:
                    features_requiring_review.append(column)

            else:
                feature_result["drift_analysis"] = None
                incompatible_features.append(column)

            feature_analysis[column] = feature_result

        review_required = bool(
            features_requiring_review
            or incompatible_features
            or alignment["reference_only_columns"]
            or alignment["current_only_columns"]
        )

        return {
            "numeric_mean_shift_threshold": (
                self.numeric_mean_shift_threshold
            ),
            "categorical_distribution_threshold": (
                self.categorical_distribution_threshold
            ),
            "column_alignment": alignment,
            "features_analyzed": len(
                alignment["shared_columns"]
            ),
            "feature_analysis": feature_analysis,
            "features_requiring_review": (
                features_requiring_review
            ),
            "incompatible_features": incompatible_features,
            "review_required": review_required,
        }
