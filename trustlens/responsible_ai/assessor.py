"""
Core Responsible AI assessor for TrustLens AI.

This module provides the central integration point for evaluating
responsible AI considerations associated with datasets and AI systems.

The assessor coordinates structural Responsible AI context with
specialised analysis components such as group fairness. Additional
bias, performance-fairness, and explainability capabilities can be
integrated as Phase 5 develops.
"""

from typing import Any, Dict, Hashable, List

import pandas as pd

from .fairness import FairnessAnalyzer


class ResponsibleAIAssessor:
    """
    Perform integrated Responsible AI assessment.

    The assessor stores dataset and modelling context and coordinates
    specialised Responsible AI analyses.

    Group fairness analysis is performed when both model predictions
    and at least one sensitive attribute are available. Each configured
    sensitive attribute is analysed independently.

    Fairness indicators identify measurable group disparities that may
    warrant review. They do not independently establish whether a model
    is fair, discriminatory, ethical, or legally compliant.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
        prediction_column: str | None = None,
        sensitive_attributes: List[str] | None = None,
        positive_label: Hashable = 1,
        disparate_impact_threshold: float = 0.80,
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
            potential bias may be assessed.
        positive_label:
            Prediction value treated as the positive outcome during
            group fairness analysis.
        disparate_impact_threshold:
            Ratio below which a group fairness comparison is flagged
            for review. This is used as a screening indicator rather
            than a universal fairness determination.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame or sensitive_attributes
            is not a list when supplied.
        ValueError
            If the dataset is empty, supplied columns do not exist, or
            the disparate-impact threshold is outside the valid range.
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

        if not 0 < disparate_impact_threshold <= 1:
            raise ValueError(
                "disparate_impact_threshold must be greater than 0 "
                "and less than or equal to 1"
            )

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
        self.positive_label = positive_label
        self.disparate_impact_threshold = float(
            disparate_impact_threshold
        )

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
        Report which Responsible AI analyses have enough context.

        Availability indicates whether minimum structural inputs are
        present. It does not indicate that an analysis has established
        fairness, bias, discrimination, or compliance.
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

    def group_fairness_analysis(self) -> Dict[str, Any]:
        """
        Run group fairness analysis for each sensitive attribute.

        Returns a not-applicable result when model predictions or
        sensitive attributes have not been supplied.
        """
        if self.prediction_column is None:
            return {
                "applicable": False,
                "reason": (
                    "Group fairness analysis requires a prediction column."
                ),
                "attributes": {},
                "review_required": False,
                "attributes_requiring_review": [],
            }

        if not self.sensitive_attributes:
            return {
                "applicable": False,
                "reason": (
                    "Group fairness analysis requires at least one "
                    "sensitive attribute."
                ),
                "attributes": {},
                "review_required": False,
                "attributes_requiring_review": [],
            }

        attribute_results: Dict[str, Any] = {}

        for attribute in self.sensitive_attributes:
            analyzer = FairnessAnalyzer(
                self.data,
                sensitive_attribute=attribute,
                prediction_column=self.prediction_column,
                positive_label=self.positive_label,
                disparate_impact_threshold=(
                    self.disparate_impact_threshold
                ),
            )

            attribute_results[attribute] = analyzer.assess()

        attributes_requiring_review = [
            attribute
            for attribute, result in attribute_results.items()
            if result["review_required"]
        ]

        return {
            "applicable": True,
            "positive_label": self.positive_label,
            "disparate_impact_threshold": (
                self.disparate_impact_threshold
            ),
            "attributes_analyzed": len(attribute_results),
            "attributes": attribute_results,
            "attributes_requiring_review": (
                attributes_requiring_review
            ),
            "review_required": bool(
                attributes_requiring_review
            ),
        }

    def responsible_ai_summary(
        self,
        group_fairness: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the current integrated Responsible AI summary.

        The summary records review indicators from implemented
        Responsible AI components. Future Phase 5 components can add
        their own indicators to this summary as they are integrated.
        """
        review_reasons: List[str] = []

        if (
            group_fairness.get("applicable", False)
            and group_fairness.get("review_required", False)
        ):
            review_reasons.append("group_fairness")

        status = "review" if review_reasons else "no_review_indicators"

        return {
            "status": status,
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "group_fairness_applicable": group_fairness.get(
                "applicable",
                False,
            ),
            "fairness_attributes_requiring_review": (
                group_fairness.get(
                    "attributes_requiring_review",
                    [],
                )
            ),
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the integrated Responsible AI assessment.

        Returns
        -------
        dict
            Responsible AI context, sensitive-attribute information,
            analysis availability, group fairness results, and an
            integrated review summary.
        """
        group_fairness = self.group_fairness_analysis()

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
            "group_fairness": group_fairness,
            "responsible_ai_summary": (
                self.responsible_ai_summary(group_fairness)
            ),
        }
