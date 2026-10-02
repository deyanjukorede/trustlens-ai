"""
Data leakage risk detection for TrustLens AI.

This module identifies structural indicators that may suggest data
leakage or inappropriate modelling features. Leakage can occur when
information unavailable at prediction time is accidentally included
in model training data, producing unrealistically strong performance.
"""

from typing import Any, Dict, List

import pandas as pd


class LeakageRiskAnalyzer:
    """
    Analyse a dataset for potential feature-to-target leakage risks.

    The analyzer focuses on explainable structural warning signals.
    These warnings do not prove that leakage exists; they identify
    columns that should receive further review before modelling.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        correlation_threshold: float = 0.95,
        high_cardinality_threshold: float = 0.90,
    ) -> None:
        """
        Initialise the leakage risk analyzer.

        Parameters
        ----------
        data:
            Dataset containing modelling features and the target.
        target_column:
            Name of the intended prediction target.
        correlation_threshold:
            Absolute correlation above which a numeric feature is
            considered suspiciously related to a numeric target.
        high_cardinality_threshold:
            Unique-value proportion above which a feature may behave
            like an identifier.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame.
        ValueError
            If the dataset is empty, target does not exist, or a
            threshold is outside its valid range.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if target_column not in data.columns:
            raise ValueError(
                f"target column '{target_column}' does not exist in the dataset"
            )

        if not 0 < correlation_threshold <= 1:
            raise ValueError(
                "correlation_threshold must be greater than 0 "
                "and less than or equal to 1"
            )

        if not 0 < high_cardinality_threshold <= 1:
            raise ValueError(
                "high_cardinality_threshold must be greater than 0 "
                "and less than or equal to 1"
            )

        self.data = data
        self.target_column = target_column
        self.correlation_threshold = float(correlation_threshold)
        self.high_cardinality_threshold = float(
            high_cardinality_threshold
        )

    @property
    def feature_columns(self) -> List[str]:
        """Return modelling features excluding the target column."""
        return [
            str(column)
            for column in self.data.columns
            if column != self.target_column
        ]

    def duplicate_target_features(self) -> List[str]:
        """
        Identify features that exactly duplicate the target.

        Exact target copies are a strong structural leakage signal.
        """
        target = self.data[self.target_column]
        duplicates: List[str] = []

        for column in self.feature_columns:
            feature = self.data[column]

            if feature.equals(target):
                duplicates.append(column)

        return duplicates

    def high_correlation_features(self) -> Dict[str, float]:
        """
        Identify numeric features highly correlated with a numeric target.

        Correlation is treated as a warning signal rather than proof of
        leakage because legitimate predictors may also be highly related
        to the target.
        """
        target = self.data[self.target_column]

        if not pd.api.types.is_numeric_dtype(target):
            return {}

        suspicious: Dict[str, float] = {}

        for column in self.feature_columns:
            feature = self.data[column]

            if not pd.api.types.is_numeric_dtype(feature):
                continue

            valid = pd.concat(
                [feature, target],
                axis=1,
            ).dropna()

            if len(valid) < 2:
                continue

            feature_values = valid.iloc[:, 0]
            target_values = valid.iloc[:, 1]

            if feature_values.nunique() <= 1:
                continue

            if target_values.nunique() <= 1:
                continue

            correlation = feature_values.corr(target_values)

            if pd.isna(correlation):
                continue

            absolute_correlation = abs(float(correlation))

            if absolute_correlation >= self.correlation_threshold:
                suspicious[column] = round(absolute_correlation, 4)

        return suspicious

    def identifier_like_features(self) -> Dict[str, float]:
        """
        Identify features with unusually high cardinality.

        Columns with nearly unique values may represent identifiers and
        can create memorisation or leakage risks in some modelling tasks.
        """
        row_count = len(self.data)

        if row_count == 0:
            return {}

        suspicious: Dict[str, float] = {}

        for column in self.feature_columns:
            feature = self.data[column]

            non_missing_count = int(feature.notna().sum())

            if non_missing_count == 0:
                continue

            unique_count = int(feature.nunique(dropna=True))

            cardinality_ratio = unique_count / non_missing_count

            if cardinality_ratio >= self.high_cardinality_threshold:
                suspicious[column] = round(cardinality_ratio, 4)

        return suspicious

    def suspicious_name_features(self) -> List[str]:
        """
        Identify feature names that may indicate target-derived data.

        This is a heuristic warning only. A flagged name requires human
        review to determine whether the feature genuinely leaks future
        or target information.
        """
        target_name = self.target_column.lower()

        leakage_terms = {
            "target",
            "label",
            "outcome",
            "result",
            "prediction",
            "predicted",
            "actual",
            "final",
        }

        suspicious: List[str] = []

        for column in self.feature_columns:
            column_name = column.lower()

            target_reference = (
                target_name in column_name
                and column_name != target_name
            )

            term_reference = any(
                term in column_name
                for term in leakage_terms
            )

            if target_reference or term_reference:
                suspicious.append(column)

        return suspicious

    def risk_features(self) -> List[str]:
        """
        Return all unique features associated with leakage warnings.
        """
        features = set(self.duplicate_target_features())
        features.update(self.high_correlation_features().keys())
        features.update(self.identifier_like_features().keys())
        features.update(self.suspicious_name_features())

        return sorted(features)

    def risk_level(self) -> str:
        """
        Return a simple descriptive leakage-risk level.

        High:
            At least one feature exactly duplicates the target.

        Medium:
            Other structural leakage warnings are present.

        Low:
            No structural warning signals are detected.
        """
        if self.duplicate_target_features():
            return "high"

        if self.risk_features():
            return "medium"

        return "low"

    def assess(self) -> Dict[str, Any]:
        """
        Run the complete TrustLens leakage-risk assessment.

        Returns
        -------
        dict
            Structured leakage warning information.
        """
        duplicate_features = self.duplicate_target_features()
        correlated_features = self.high_correlation_features()
        identifier_features = self.identifier_like_features()
        suspicious_names = self.suspicious_name_features()
        risk_features = self.risk_features()

        return {
            "target_column": self.target_column,
            "features_analyzed": len(self.feature_columns),
            "duplicate_target_features": duplicate_features,
            "high_correlation_features": correlated_features,
            "identifier_like_features": identifier_features,
            "suspicious_name_features": suspicious_names,
            "risk_features": risk_features,
            "risk_feature_count": len(risk_features),
            "risk_level": self.risk_level(),
            "correlation_threshold": self.correlation_threshold,
            "high_cardinality_threshold": (
                self.high_cardinality_threshold
            ),
        }
