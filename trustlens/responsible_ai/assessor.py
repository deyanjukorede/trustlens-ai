"""
Core Responsible AI assessor for TrustLens AI.

This module provides the foundation for evaluating responsible AI
considerations associated with datasets and AI systems.

The assessor is designed to coordinate specialised TrustLens
components for fairness, potential bias, explainability, and other
responsible AI indicators as Phase 5 develops.
"""

from typing import Any, Dict, List

import pandas as pd


class ResponsibleAIAssessor:
    """
    Perform foundational Responsible AI assessment.

    The assessor stores the dataset and optional modelling context
    required by specialised Responsible AI components.

    Parameters such as a target column, prediction column, and
    sensitive attributes are optional because some Responsible AI
    checks can be performed without all modelling outputs being
    available.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
        prediction_column: str | None = None,
        sensitive_attributes: List[str] | None = None,
    ) -> None:
        """
        Initialise the Responsible AI assessor.

        Parameters
        ----------
        data:
            Dataset to be assessed.
        target_column:
            Optional column containing observed or expected outcomes.
        prediction_column:
            Optional column containing model predictions.
        sensitive_attributes:
            Optional list of attributes across which fairness and
            potential bias may later be assessed.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame or sensitive_attributes
            is not a list when supplied.
        ValueError
            If the dataset is empty or supplied columns do not exist.
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

        self.data = data
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.sensitive_attributes = sensitive_attributes

    @property
    def row_count(self) -> int:
        """Return the number of rows in the dataset."""
        return int(self.data.shape[0])

    @property
    def column_count(self) -> int:
        """Return the number of columns in the dataset."""
        return int(self.data.shape[1])

    @property
    def sensitive_attribute_count(self) -> int:
        """Return the number of configured sensitive attributes."""
        return len(self.sensitive_attributes)

    def assessment_context(self) -> Dict[str, Any]:
        """
        Return the Responsible AI assessment context.

        This describes which modelling and sensitive-attribute
        information is currently available for downstream analyses.
        """
        return {
            "target_available": self.target_column is not None,
            "predictions_available": self.prediction_column is not None,
            "sensitive_attributes_available": (
                self.sensitive_attribute_count > 0
            ),
            "sensitive_attribute_count": self.sensitive_attribute_count,
        }

    def sensitive_attribute_summary(self) -> Dict[str, Any]:
        """
        Summarise configured sensitive attributes.

        The method reports structural information only. It does not
        determine whether an attribute is legally protected or whether
        observed differences constitute unfair treatment.
        """
        attributes: Dict[str, Any] = {}

        for attribute in self.sensitive_attributes:
            series = self.data[attribute]

            value_counts = series.value_counts(
                dropna=False
            ).to_dict()

            attributes[attribute] = {
                "unique_values": int(series.nunique(dropna=True)),
                "missing_values": int(series.isna().sum()),
                "group_counts": {
                    str(group): int(count)
                    for group, count in value_counts.items()
                },
            }

        return {
            "attribute_count": self.sensitive_attribute_count,
            "attributes": attributes,
        }

    def analysis_availability(self) -> Dict[str, Any]:
        """
        Report which future Responsible AI analyses have enough context.

        Availability indicates only whether the minimum structural
        inputs are present. It does not indicate that an analysis has
        already been performed.
        """
        has_sensitive_attributes = self.sensitive_attribute_count > 0
        has_target = self.target_column is not None
        has_predictions = self.prediction_column is not None

        return {
            "group_fairness": {
                "available": (
                    has_sensitive_attributes and has_predictions
                ),
                "requires": [
                    "sensitive_attributes",
                    "prediction_column",
                ],
            },
            "outcome_bias": {
                "available": (
                    has_sensitive_attributes and has_target
                ),
                "requires": [
                    "sensitive_attributes",
                    "target_column",
                ],
            },
            "prediction_performance_fairness": {
                "available": (
                    has_sensitive_attributes
                    and has_target
                    and has_predictions
                ),
                "requires": [
                    "sensitive_attributes",
                    "target_column",
                    "prediction_column",
                ],
            },
            "explainability": {
                "available": has_predictions,
                "requires": [
                    "prediction_column",
                ],
            },
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the foundational Responsible AI assessment.

        Returns
        -------
        dict
            Structural Responsible AI information that will serve as
            the integration point for fairness, bias, and explainability
            components developed during Phase 5.
        """
        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "target_column": self.target_column,
            "prediction_column": self.prediction_column,
            "sensitive_attributes": list(self.sensitive_attributes),
            "assessment_context": self.assessment_context(),
            "sensitive_attribute_summary": (
                self.sensitive_attribute_summary()
            ),
            "analysis_availability": self.analysis_availability(),
        }
