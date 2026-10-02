"""
Feature suitability assessment for TrustLens AI.

This module evaluates whether dataset features are structurally suitable
for machine-learning workflows. It identifies common readiness concerns
such as constant features, low-variation features, high-cardinality
categorical features, identifier-like columns, and excessive missingness.

The assessment is intentionally model-agnostic. It highlights potential
issues that should be reviewed before modelling rather than automatically
removing or transforming features.
"""

from typing import Any, Dict, List

import pandas as pd


class FeatureSuitabilityAnalyzer:
    """
    Assess modelling features for common AI-readiness concerns.

    Parameters
    ----------
    data:
        Dataset containing the features to assess.
    target_column:
        Optional prediction target. When supplied, the target is excluded
        from feature-suitability analysis.
    missing_threshold:
        Percentage of missing values at or above which a feature is flagged
        for excessive missingness.
    high_cardinality_threshold:
        Percentage of unique non-missing values at or above which a
        categorical feature may be considered high cardinality.
    identifier_threshold:
        Percentage of unique non-missing values at or above which a feature
        may be considered identifier-like.
    low_variation_threshold:
        Percentage represented by the most frequent non-missing value at or
        above which a non-constant feature is considered low variation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
        missing_threshold: float = 40.0,
        high_cardinality_threshold: float = 50.0,
        identifier_threshold: float = 95.0,
        low_variation_threshold: float = 95.0,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if target_column is not None and target_column not in data.columns:
            raise ValueError(
                f"target_column '{target_column}' does not exist in the dataset"
            )

        self._validate_percentage(
            missing_threshold,
            "missing_threshold",
        )
        self._validate_percentage(
            high_cardinality_threshold,
            "high_cardinality_threshold",
        )
        self._validate_percentage(
            identifier_threshold,
            "identifier_threshold",
        )
        self._validate_percentage(
            low_variation_threshold,
            "low_variation_threshold",
        )

        self.data = data
        self.target_column = target_column
        self.missing_threshold = float(missing_threshold)
        self.high_cardinality_threshold = float(high_cardinality_threshold)
        self.identifier_threshold = float(identifier_threshold)
        self.low_variation_threshold = float(low_variation_threshold)

    @staticmethod
    def _validate_percentage(value: float, name: str) -> None:
        """Validate that a threshold is between 0 and 100."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")

        if value < 0 or value > 100:
            raise ValueError(f"{name} must be between 0 and 100")

    @property
    def feature_columns(self) -> List[str]:
        """Return columns included in feature-suitability analysis."""
        columns = [str(column) for column in self.data.columns]

        if self.target_column is None:
            return columns

        return [
            column
            for column in columns
            if column != self.target_column
        ]

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        """Return whether a pandas Series is categorical-like."""
        return bool(
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

    def _column_metrics(self, column: str) -> Dict[str, Any]:
        """Calculate structural suitability metrics for one feature."""
        series = self.data[column]

        row_count = len(series)
        missing_count = int(series.isna().sum())
        non_missing_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))

        missing_rate = (
            (missing_count / row_count) * 100.0
            if row_count
            else 0.0
        )

        unique_rate = (
            (unique_count / non_missing_count) * 100.0
            if non_missing_count
            else 0.0
        )

        if non_missing_count:
            value_counts = series.dropna().value_counts()
            dominant_count = int(value_counts.iloc[0]) if not value_counts.empty else 0
            dominant_rate = (dominant_count / non_missing_count) * 100.0
        else:
            dominant_rate = 0.0

        return {
            "data_type": str(series.dtype),
            "missing_count": missing_count,
            "missing_rate": round(missing_rate, 2),
            "non_missing_count": non_missing_count,
            "unique_count": unique_count,
            "unique_rate": round(unique_rate, 2),
            "dominant_value_rate": round(dominant_rate, 2),
        }

    def analyze_feature(self, column: str) -> Dict[str, Any]:
        """
        Assess a single feature.

        Returns
        -------
        dict
            Feature metrics, detected issues, suitability status,
            and recommendations.
        """
        if column not in self.feature_columns:
            raise ValueError(
                f"column '{column}' is not available for feature analysis"
            )

        series = self.data[column]
        metrics = self._column_metrics(column)

        issues: List[str] = []
        recommendations: List[str] = []

        unique_count = metrics["unique_count"]
        missing_rate = metrics["missing_rate"]
        unique_rate = metrics["unique_rate"]
        dominant_rate = metrics["dominant_value_rate"]

        if unique_count <= 1:
            issues.append("constant_feature")
            recommendations.append(
                "Consider removing this feature because it provides "
                "no meaningful variation."
            )

        elif dominant_rate >= self.low_variation_threshold:
            issues.append("low_variation")
            recommendations.append(
                "Review this feature because one value dominates "
                "most observations."
            )

        if missing_rate >= self.missing_threshold:
            issues.append("excessive_missingness")
            recommendations.append(
                "Review missing-data handling before using this feature "
                "for modelling."
            )

        is_categorical = self._is_categorical(series)

        if (
            is_categorical
            and unique_count > 1
            and unique_rate >= self.high_cardinality_threshold
        ):
            issues.append("high_cardinality")
            recommendations.append(
                "Review encoding strategy because this categorical feature "
                "has high cardinality."
            )

        if (
            unique_count > 1
            and unique_rate >= self.identifier_threshold
        ):
            issues.append("identifier_like")
            recommendations.append(
                "Review whether this feature is an identifier rather than "
                "a meaningful predictive variable."
            )

        status = "suitable" if not issues else "review"

        return {
            "column": column,
            "status": status,
            "issues": issues,
            "recommendations": recommendations,
            **metrics,
        }

    def analyze(self) -> Dict[str, Any]:
        """
        Run feature-suitability analysis across all modelling features.

        Returns
        -------
        dict
            Dataset-level summary and per-feature suitability results.
        """
        feature_results = {
            column: self.analyze_feature(column)
            for column in self.feature_columns
        }

        features_requiring_review = [
            column
            for column, result in feature_results.items()
            if result["status"] == "review"
        ]

        suitable_features = [
            column
            for column, result in feature_results.items()
            if result["status"] == "suitable"
        ]

        issue_counts: Dict[str, int] = {}

        for result in feature_results.values():
            for issue in result["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        feature_count = len(self.feature_columns)

        suitability_rate = (
            (len(suitable_features) / feature_count) * 100.0
            if feature_count
            else 0.0
        )

        return {
            "feature_count": feature_count,
            "suitable_feature_count": len(suitable_features),
            "features_requiring_review_count": len(features_requiring_review),
            "suitability_rate": round(suitability_rate, 2),
            "suitable_features": suitable_features,
            "features_requiring_review": features_requiring_review,
            "issue_counts": issue_counts,
            "features": feature_results,
        }
