"""
Class imbalance analysis for TrustLens AI.

This module evaluates the distribution of classes in a prediction
target and identifies potential imbalance that may affect machine
learning model training, evaluation, and reliability.
"""

from typing import Any, Dict, List

import pandas as pd


class ClassImbalanceAnalyzer:
    """
    Analyse class distribution and imbalance in a target variable.

    The analyzer provides descriptive information about target classes
    and identifies whether the target distribution may require
    additional modelling attention.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        imbalance_threshold: float = 0.20,
    ) -> None:
        """
        Initialise the class imbalance analyzer.

        Parameters
        ----------
        data:
            Dataset containing the prediction target.
        target_column:
            Name of the target column to analyse.
        imbalance_threshold:
            Minimum minority-class proportion considered acceptable.
            The default value is 0.20, representing 20 percent.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame.
        ValueError
            If the dataset is empty, the target column does not exist,
            or the imbalance threshold is invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if target_column not in data.columns:
            raise ValueError(
                f"target column '{target_column}' does not exist in the dataset"
            )

        if not 0 < imbalance_threshold <= 1:
            raise ValueError(
                "imbalance_threshold must be greater than 0 and less than or equal to 1"
            )

        self.data = data
        self.target_column = target_column
        self.imbalance_threshold = float(imbalance_threshold)

    @property
    def target(self) -> pd.Series:
        """Return the target series."""
        return self.data[self.target_column]

    def class_counts(self) -> Dict[Any, int]:
        """
        Return the number of observations in each target class.

        Missing target values are excluded from class counts.
        """
        counts = self.target.dropna().value_counts()

        return {
            class_value: int(count)
            for class_value, count in counts.items()
        }

    def class_proportions(self) -> Dict[Any, float]:
        """
        Return the percentage representation of each target class.
        """
        counts = self.class_counts()
        total = sum(counts.values())

        if total == 0:
            return {}

        return {
            class_value: round((count / total) * 100.0, 2)
            for class_value, count in counts.items()
        }

    def minority_classes(self) -> List[Any]:
        """
        Return classes below the configured imbalance threshold.
        """
        counts = self.class_counts()
        total = sum(counts.values())

        if total == 0:
            return []

        return [
            class_value
            for class_value, count in counts.items()
            if (count / total) < self.imbalance_threshold
        ]

    def majority_class(self) -> Any:
        """
        Return the most frequently occurring target class.

        Returns None when no non-missing target observations exist.
        """
        counts = self.class_counts()

        if not counts:
            return None

        return max(counts, key=counts.get)

    def minority_class(self) -> Any:
        """
        Return the least frequently occurring target class.

        Returns None when no non-missing target observations exist.
        """
        counts = self.class_counts()

        if not counts:
            return None

        return min(counts, key=counts.get)

    def imbalance_ratio(self) -> float:
        """
        Calculate the majority-to-minority class ratio.

        A value of 1.0 indicates equally represented classes.
        Higher values indicate increasing imbalance.

        Returns 0.0 when no valid class distribution is available.
        """
        counts = self.class_counts()

        if not counts:
            return 0.0

        majority_count = max(counts.values())
        minority_count = min(counts.values())

        if minority_count == 0:
            return 0.0

        return round(majority_count / minority_count, 2)

    def is_imbalanced(self) -> bool:
        """
        Determine whether the target contains underrepresented classes.

        A dataset is considered imbalanced when at least one target
        class represents less than the configured threshold.
        """
        return len(self.minority_classes()) > 0

    def missing_target_summary(self) -> Dict[str, Any]:
        """
        Summarise missing values in the target column.
        """
        missing_count = int(self.target.isna().sum())
        total_rows = int(len(self.target))

        missing_rate = (
            (missing_count / total_rows) * 100.0
            if total_rows
            else 0.0
        )

        return {
            "missing_count": missing_count,
            "missing_rate": round(missing_rate, 2),
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the complete TrustLens class imbalance assessment.

        Returns
        -------
        dict
            Structured class-distribution and imbalance information.
        """
        counts = self.class_counts()
        proportions = self.class_proportions()

        return {
            "target_column": self.target_column,
            "class_count": len(counts),
            "class_counts": counts,
            "class_proportions": proportions,
            "majority_class": self.majority_class(),
            "minority_class": self.minority_class(),
            "minority_classes": self.minority_classes(),
            "imbalance_ratio": self.imbalance_ratio(),
            "imbalance_threshold": self.imbalance_threshold,
            "is_imbalanced": self.is_imbalanced(),
            "missing_target": self.missing_target_summary(),
        }
