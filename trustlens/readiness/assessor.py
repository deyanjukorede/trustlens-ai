"""
Core AI readiness assessor for TrustLens AI.

This module provides the integrated foundation for evaluating whether
datasets are structurally suitable for artificial intelligence and
machine learning workflows.

The assessor combines foundational dataset readiness information with
feature suitability, class imbalance, and data leakage risk analysis.
"""

from typing import Any, Dict, List

import pandas as pd

from .class_imbalance import ClassImbalanceAnalyzer
from .feature_suitability import FeatureSuitabilityAnalyzer
from ..leakage import LeakageRiskAnalyzer


class AIReadinessAssessor:
    """
    Perform integrated AI readiness assessment of a pandas DataFrame.

    The assessor evaluates structural characteristics of a dataset and
    coordinates specialised TrustLens readiness components for feature
    suitability, target-class imbalance, and potential data leakage.

    Class-imbalance and leakage analysis require a target column.
    When no target column is supplied, those analyses are reported as
    not applicable rather than causing the overall assessment to fail.
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

    def feature_suitability_analysis(self) -> Dict[str, Any]:
        """
        Run feature-suitability analysis.

        The prediction target, when supplied, is excluded from feature
        suitability assessment by the specialised analyzer.
        """
        analyzer = FeatureSuitabilityAnalyzer(
            self.data,
            target_column=self.target_column,
        )
        return analyzer.analyze()

    def class_imbalance_analysis(self) -> Dict[str, Any]:
        """
        Run target-class imbalance analysis.

        Returns a not-applicable result when no prediction target has
        been supplied.
        """
        if self.target_column is None:
            return {
                "applicable": False,
                "reason": (
                    "Class imbalance analysis requires a target column."
                ),
            }

        analyzer = ClassImbalanceAnalyzer(
            self.data,
            target_column=self.target_column,
        )

        result = analyzer.assess()
        result["applicable"] = True

        return result

    def leakage_risk_analysis(self) -> Dict[str, Any]:
        """
        Run structural data-leakage risk analysis.

        Returns a not-applicable result when no prediction target has
        been supplied.
        """
        if self.target_column is None:
            return {
                "applicable": False,
                "reason": (
                    "Data leakage analysis requires a target column."
                ),
            }

        analyzer = LeakageRiskAnalyzer(
            self.data,
            target_column=self.target_column,
        )

        result = analyzer.assess()
        result["applicable"] = True

        return result

    def readiness_summary(
        self,
        feature_suitability: Dict[str, Any],
        class_imbalance: Dict[str, Any],
        leakage_risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a concise integrated AI-readiness summary.

        This summary does not claim that a dataset is automatically
        safe or appropriate for modelling. It surfaces structural
        conditions that may require review before model development.
        """
        features_requiring_review = feature_suitability.get(
            "features_requiring_review",
            [],
        )

        class_imbalance_detected = (
            class_imbalance.get("is_imbalanced", False)
            if class_imbalance.get("applicable", False)
            else None
        )

        leakage_level = (
            leakage_risk.get("risk_level")
            if leakage_risk.get("applicable", False)
            else None
        )

        review_reasons: List[str] = []

        if features_requiring_review:
            review_reasons.append("feature_suitability")

        if class_imbalance_detected is True:
            review_reasons.append("class_imbalance")

        if leakage_level in {"medium", "high"}:
            review_reasons.append("data_leakage")

        status = "review" if review_reasons else "ready"

        return {
            "status": status,
            "review_reasons": review_reasons,
            "features_requiring_review": len(features_requiring_review),
            "class_imbalance_detected": class_imbalance_detected,
            "leakage_risk_level": leakage_level,
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the integrated TrustLens AI readiness assessment.

        Returns
        -------
        dict
            Structured readiness information combining foundational
            dataset information with feature suitability, class
            imbalance, leakage risk, and an integrated summary.
        """
        feature_suitability = self.feature_suitability_analysis()
        class_imbalance = self.class_imbalance_analysis()
        leakage_risk = self.leakage_risk_analysis()

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
            "feature_suitability": feature_suitability,
            "class_imbalance": class_imbalance,
            "leakage_risk": leakage_risk,
            "readiness_summary": self.readiness_summary(
                feature_suitability,
                class_imbalance,
                leakage_risk,
            ),
        }
