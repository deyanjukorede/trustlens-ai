"""
Explainability analysis for TrustLens AI.

This module provides structural indicators for assessing whether a
dataset contains suitable feature information to support meaningful
model explanation.

The analysis focuses on explanation readiness rather than attempting
to prove that a model is explainable. Model-specific explanation
methods can be added in future versions of TrustLens.
"""

from typing import Any, Dict, List

import pandas as pd


class ExplainabilityAnalyzer:
    """
    Assess feature-level readiness for model explainability.

    The analyzer identifies candidate explanatory features and reports
    structural conditions that may make model explanations more
    difficult to interpret or generate reliably.

    It does not determine whether a model is inherently explainable,
    transparent, ethical, or compliant.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
        prediction_column: str | None = None,
        sensitive_attributes: List[str] | None = None,
        high_cardinality_threshold: int = 20,
        missingness_threshold: float = 0.50,
    ) -> None:
        """
        Initialise the explainability analyzer.

        Parameters
        ----------
        data:
            Dataset containing candidate explanatory features.
        target_column:
            Optional observed-outcome column to exclude from explanatory
            feature analysis.
        prediction_column:
            Optional model-prediction column to exclude from explanatory
            feature analysis.
        sensitive_attributes:
            Optional sensitive attributes to exclude from candidate
            explanatory features.
        high_cardinality_threshold:
            Number of unique values above which a categorical feature
            is flagged as high cardinality.
        missingness_threshold:
            Proportion of missing values at or above which a feature is
            flagged for explainability review.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame or sensitive_attributes
            is not a list when supplied.
        ValueError
            If data is empty, configured columns are missing, or
            thresholds are invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if target_column is not None and target_column not in data.columns:
            raise ValueError(
                f"target_column '{target_column}' does not exist in the dataset"
            )

        if (
            prediction_column is not None
            and prediction_column not in data.columns
        ):
            raise ValueError(
                "prediction_column "
                f"'{prediction_column}' does not exist in the dataset"
            )

        if (
            sensitive_attributes is not None
            and not isinstance(sensitive_attributes, list)
        ):
            raise TypeError("sensitive_attributes must be a list")

        sensitive_attributes = sensitive_attributes or []

        missing_sensitive_attributes = [
            attribute
            for attribute in sensitive_attributes
            if attribute not in data.columns
        ]

        if missing_sensitive_attributes:
            raise ValueError(
                "The following sensitive attributes do not exist "
                "in the dataset: "
                f"{missing_sensitive_attributes}"
            )

        if (
            not isinstance(high_cardinality_threshold, int)
            or isinstance(high_cardinality_threshold, bool)
            or high_cardinality_threshold < 1
        ):
            raise ValueError(
                "high_cardinality_threshold must be a positive integer"
            )

        if not 0 <= missingness_threshold <= 1:
            raise ValueError(
                "missingness_threshold must be between 0 and 1"
            )

        self.data = data
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.sensitive_attributes = sensitive_attributes

        self.high_cardinality_threshold = (
            high_cardinality_threshold
        )

        self.missingness_threshold = float(
            missingness_threshold
        )

    def excluded_columns(self) -> List[str]:
        """
        Return columns excluded from explanatory feature analysis.

        Target, prediction, and configured sensitive attributes are
        excluded when present.
        """
        excluded: List[str] = []

        if self.target_column is not None:
            excluded.append(self.target_column)

        if self.prediction_column is not None:
            excluded.append(self.prediction_column)

        for attribute in self.sensitive_attributes:
            if attribute not in excluded:
                excluded.append(attribute)

        return excluded

    def candidate_features(self) -> List[str]:
        """Return columns available as candidate explanatory features."""
        excluded = set(self.excluded_columns())

        return [
            column
            for column in self.data.columns
            if column not in excluded
        ]

    def feature_type(self, series: pd.Series) -> str:
        """Return a simplified feature-type classification."""
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return "categorical"

        return "other"

    def feature_analysis(self) -> Dict[str, Any]:
        """
        Analyse candidate explanatory features.

        Each feature is assessed for type, missingness, uniqueness,
        constant values, and categorical cardinality.
        """
        results: Dict[str, Any] = {}

        for feature in self.candidate_features():
            series = self.data[feature]

            feature_type = self.feature_type(series)

            missing_count = int(series.isna().sum())

            missing_rate = (
                missing_count / len(series)
                if len(series) > 0
                else 0.0
            )

            unique_values = int(
                series.nunique(dropna=True)
            )

            constant_feature = unique_values <= 1

            high_missingness = (
                missing_rate >= self.missingness_threshold
            )

            high_cardinality = (
                feature_type == "categorical"
                and unique_values
                > self.high_cardinality_threshold
            )

            indicators: List[str] = []

            if constant_feature:
                indicators.append("constant_feature")

            if high_missingness:
                indicators.append("high_missingness")

            if high_cardinality:
                indicators.append("high_cardinality")

            results[feature] = {
                "feature_type": feature_type,
                "missing_count": missing_count,
                "missing_rate": round(missing_rate, 4),
                "unique_values": unique_values,
                "constant_feature": constant_feature,
                "high_missingness": high_missingness,
                "high_cardinality": high_cardinality,
                "indicators": indicators,
                "requires_review": bool(indicators),
            }

        return results

    def feature_type_summary(
        self,
        feature_results: Dict[str, Any],
    ) -> Dict[str, int]:
        """Summarise candidate features by simplified feature type."""
        summary = {
            "numeric": 0,
            "categorical": 0,
            "boolean": 0,
            "datetime": 0,
            "other": 0,
        }

        for result in feature_results.values():
            feature_type = result["feature_type"]

            summary[feature_type] = (
                summary.get(feature_type, 0) + 1
            )

        return summary

    def review_summary(
        self,
        feature_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Summarise feature-level explainability review indicators.
        """
        constant_features = [
            feature
            for feature, result in feature_results.items()
            if result["constant_feature"]
        ]

        high_missingness_features = [
            feature
            for feature, result in feature_results.items()
            if result["high_missingness"]
        ]

        high_cardinality_features = [
            feature
            for feature, result in feature_results.items()
            if result["high_cardinality"]
        ]

        features_requiring_review = [
            feature
            for feature, result in feature_results.items()
            if result["requires_review"]
        ]

        return {
            "constant_features": constant_features,
            "high_missingness_features": high_missingness_features,
            "high_cardinality_features": high_cardinality_features,
            "features_requiring_review": features_requiring_review,
            "review_required": bool(features_requiring_review),
        }

    def readiness_summary(
        self,
        feature_results: Dict[str, Any],
        review: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a structural explanation-readiness summary.

        The result describes whether candidate explanatory features are
        available and whether structural indicators warrant review.
        It does not certify model explainability.
        """
        candidate_count = len(feature_results)

        usable_features = [
            feature
            for feature, result in feature_results.items()
            if not result["constant_feature"]
            and not result["high_missingness"]
        ]

        if candidate_count == 0:
            status = "insufficient_features"
            review_required = True

        elif not usable_features:
            status = "limited_explanation_readiness"
            review_required = True

        elif review["review_required"]:
            status = "review"
            review_required = True

        else:
            status = "ready_for_explanation_analysis"
            review_required = False

        return {
            "status": status,
            "candidate_feature_count": candidate_count,
            "usable_feature_count": len(usable_features),
            "usable_features": usable_features,
            "review_required": review_required,
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the complete explainability-readiness assessment.

        Returns
        -------
        dict
            Candidate-feature information, structural feature analysis,
            feature-type summary, review indicators, and explanation
            readiness.
        """
        excluded = self.excluded_columns()
        candidates = self.candidate_features()
        feature_results = self.feature_analysis()

        review = self.review_summary(
            feature_results
        )

        readiness = self.readiness_summary(
            feature_results,
            review,
        )

        return {
            "target_column": self.target_column,
            "prediction_column": self.prediction_column,
            "sensitive_attributes": list(
                self.sensitive_attributes
            ),
            "high_cardinality_threshold": (
                self.high_cardinality_threshold
            ),
            "missingness_threshold": (
                self.missingness_threshold
            ),
            "excluded_columns": excluded,
            "candidate_features": candidates,
            "candidate_feature_count": len(candidates),
            "feature_types": self.feature_type_summary(
                feature_results
            ),
            "feature_analysis": feature_results,
            "review_summary": review,
            "explanation_readiness": readiness,
            "review_required": readiness[
                "review_required"
            ],
        }
