"""
Data stability analysis for TrustLens AI.

This module evaluates structural and operational characteristics that
may affect the ongoing stability of a dataset.

The analysis focuses on measurable indicators such as missingness
concentration, constant and near-constant features, uniqueness,
high-cardinality categorical features, duplicate pressure, and
structural characteristics.

These indicators are intended to support investigation and human
review. They do not independently establish that a dataset or AI
system is stable, unstable, reliable, or unreliable.
"""

from typing import Any, Dict, List

import pandas as pd


class DataStabilityAnalyzer:
    """
    Analyse indicators related to dataset stability.

    Parameters
    ----------
    data:
        Current dataset being assessed.
    high_missingness_threshold:
        Missing-value rate above which a feature requires review.
    near_constant_threshold:
        Dominant-value proportion above which a non-constant feature
        is considered near-constant.
    high_cardinality_threshold:
        Unique-value proportion above which a categorical feature may
        require review.
    duplicate_rate_threshold:
        Duplicate-row rate above which duplicate pressure requires
        review.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        high_missingness_threshold: float = 0.50,
        near_constant_threshold: float = 0.95,
        high_cardinality_threshold: float = 0.90,
        duplicate_rate_threshold: float = 0.10,
    ) -> None:
        """Initialise the data stability analyzer."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

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

    @staticmethod
    def _validate_threshold(
        name: str,
        value: float,
    ) -> None:
        """Validate a proportion-based threshold."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")

        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

    def _missingness_analysis(
        self,
        column: str,
    ) -> Dict[str, Any]:
        """Analyse missing-value concentration for one feature."""
        missing_count = int(
            self.data[column].isna().sum()
        )

        missing_rate = round(
            missing_count / self.row_count,
            4,
        )

        return {
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "threshold": self.high_missingness_threshold,
            "requires_review": (
                missing_rate > self.high_missingness_threshold
            ),
        }

    def _concentration_analysis(
        self,
        column: str,
    ) -> Dict[str, Any]:
        """Analyse constant and near-constant feature behaviour."""
        non_missing = self.data[column].dropna()

        if non_missing.empty:
            return {
                "unique_non_missing_values": 0,
                "dominant_value_rate": None,
                "is_constant": False,
                "is_near_constant": False,
                "threshold": self.near_constant_threshold,
                "requires_review": False,
            }

        value_counts = non_missing.value_counts(
            normalize=True,
            dropna=True,
        )

        unique_count = int(non_missing.nunique())

        dominant_value_rate = round(
            float(value_counts.iloc[0]),
            4,
        )

        is_constant = unique_count == 1

        is_near_constant = (
            not is_constant
            and dominant_value_rate
            > self.near_constant_threshold
        )

        return {
            "unique_non_missing_values": unique_count,
            "dominant_value_rate": dominant_value_rate,
            "is_constant": is_constant,
            "is_near_constant": is_near_constant,
            "threshold": self.near_constant_threshold,
            "requires_review": (
                is_constant or is_near_constant
            ),
        }

    def _cardinality_analysis(
        self,
        column: str,
    ) -> Dict[str, Any]:
        """Analyse uniqueness and categorical cardinality."""
        non_missing = self.data[column].dropna()

        non_missing_count = int(len(non_missing))

        unique_count = int(
            non_missing.nunique()
        )

        if non_missing_count == 0:
            unique_rate = 0.0
        else:
            unique_rate = round(
                unique_count / non_missing_count,
                4,
            )

        is_categorical = (
            not pd.api.types.is_numeric_dtype(
                self.data[column]
            )
            and not pd.api.types.is_datetime64_any_dtype(
                self.data[column]
            )
        )

        high_cardinality = (
            is_categorical
            and non_missing_count > 1
            and unique_count > 1
            and unique_rate
            > self.high_cardinality_threshold
        )

        return {
            "unique_count": unique_count,
            "non_missing_count": non_missing_count,
            "unique_rate": unique_rate,
            "is_categorical": is_categorical,
            "high_cardinality": high_cardinality,
            "threshold": self.high_cardinality_threshold,
            "requires_review": high_cardinality,
        }

    def _duplicate_analysis(self) -> Dict[str, Any]:
        """Analyse duplicate-row pressure across the dataset."""
        duplicate_count = int(
            self.data.duplicated().sum()
        )

        duplicate_rate = round(
            duplicate_count / self.row_count,
            4,
        )

        return {
            "duplicate_rows": duplicate_count,
            "duplicate_rate": duplicate_rate,
            "threshold": self.duplicate_rate_threshold,
            "requires_review": (
                duplicate_rate > self.duplicate_rate_threshold
            ),
        }

    def _dtype_summary(self) -> Dict[str, int]:
        """Summarise broad feature-type composition."""
        numeric = 0
        datetime = 0
        categorical = 0

        for column in self.data.columns:
            series = self.data[column]

            if pd.api.types.is_numeric_dtype(series):
                numeric += 1
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetime += 1
            else:
                categorical += 1

        return {
            "numeric_features": numeric,
            "categorical_features": categorical,
            "datetime_features": datetime,
        }

    def assess(self) -> Dict[str, Any]:
        """Return the complete data stability assessment."""
        feature_analysis: Dict[str, Dict[str, Any]] = {}

        high_missingness_features: List[str] = []
        constant_features: List[str] = []
        near_constant_features: List[str] = []
        high_cardinality_features: List[str] = []

        for column in self.data.columns:
            missingness = self._missingness_analysis(
                column
            )
            concentration = self._concentration_analysis(
                column
            )
            cardinality = self._cardinality_analysis(
                column
            )

            if missingness["requires_review"]:
                high_missingness_features.append(column)

            if concentration["is_constant"]:
                constant_features.append(column)

            if concentration["is_near_constant"]:
                near_constant_features.append(column)

            if cardinality["requires_review"]:
                high_cardinality_features.append(column)

            feature_analysis[column] = {
                "missingness": missingness,
                "concentration": concentration,
                "cardinality": cardinality,
            }

        duplicates = self._duplicate_analysis()

        review_reasons: List[str] = []

        if high_missingness_features:
            review_reasons.append("high_missingness")

        if constant_features:
            review_reasons.append("constant_features")

        if near_constant_features:
            review_reasons.append("near_constant_features")

        if high_cardinality_features:
            review_reasons.append("high_cardinality")

        if duplicates["requires_review"]:
            review_reasons.append("duplicate_pressure")

        review_required = bool(review_reasons)

        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "thresholds": {
                "high_missingness": (
                    self.high_missingness_threshold
                ),
                "near_constant": (
                    self.near_constant_threshold
                ),
                "high_cardinality": (
                    self.high_cardinality_threshold
                ),
                "duplicate_rate": (
                    self.duplicate_rate_threshold
                ),
            },
            "feature_type_summary": self._dtype_summary(),
            "feature_analysis": feature_analysis,
            "high_missingness_features": (
                high_missingness_features
            ),
            "constant_features": constant_features,
            "near_constant_features": (
                near_constant_features
            ),
            "high_cardinality_features": (
                high_cardinality_features
            ),
            "duplicate_analysis": duplicates,
            "review_required": review_required,
            "review_reasons": review_reasons,
        }
