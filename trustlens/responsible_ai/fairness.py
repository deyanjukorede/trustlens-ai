"""
Group fairness analysis for TrustLens AI.

This module provides descriptive fairness indicators for comparing
model outcomes across groups defined by a sensitive attribute.

The analyzer reports measurable disparities such as selection rates,
rate differences, and disparate-impact ratios. These indicators are
intended to support human review and should not, by themselves, be
interpreted as proof that a model is fair, unfair, discriminatory,
or legally compliant.
"""

from typing import Any, Dict, Hashable

import pandas as pd


class FairnessAnalyzer:
    """
    Analyse group-level outcome disparities.

    The analyzer compares positive prediction rates across groups
    defined by a sensitive attribute.

    Parameters
    ----------
    data:
        Dataset containing the sensitive attribute and prediction.
    sensitive_attribute:
        Column used to define groups for comparison.
    prediction_column:
        Column containing model predictions.
    positive_label:
        Prediction value treated as the positive outcome.
    reference_group:
        Optional group against which other groups are compared.
        When omitted, the group with the highest selection rate is
        used as the reference group.
    disparate_impact_threshold:
        Ratio below which a group is flagged for review. The default
        value of 0.80 is a commonly used screening threshold and is
        not treated as a universal definition of fairness.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sensitive_attribute: str,
        prediction_column: str,
        positive_label: Hashable = 1,
        reference_group: Hashable | None = None,
        disparate_impact_threshold: float = 0.80,
    ) -> None:
        """Initialise the group fairness analyzer."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if sensitive_attribute not in data.columns:
            raise ValueError(
                "sensitive_attribute "
                f"'{sensitive_attribute}' does not exist in the dataset"
            )

        if prediction_column not in data.columns:
            raise ValueError(
                "prediction_column "
                f"'{prediction_column}' does not exist in the dataset"
            )

        if not 0 < disparate_impact_threshold <= 1:
            raise ValueError(
                "disparate_impact_threshold must be greater than 0 "
                "and less than or equal to 1"
            )

        if reference_group is not None:
            available_groups = set(
                data[sensitive_attribute].dropna().unique().tolist()
            )

            if reference_group not in available_groups:
                raise ValueError(
                    f"reference_group '{reference_group}' does not exist "
                    f"in sensitive_attribute '{sensitive_attribute}'"
                )

        self.data = data
        self.sensitive_attribute = sensitive_attribute
        self.prediction_column = prediction_column
        self.positive_label = positive_label
        self.reference_group = reference_group
        self.disparate_impact_threshold = (
            float(disparate_impact_threshold)
        )

    def valid_data(self) -> pd.DataFrame:
        """
        Return rows with non-missing group and prediction values.

        Missing sensitive-group or prediction values are excluded from
        rate calculations and reported separately in the final result.
        """
        return self.data[
            self.data[self.sensitive_attribute].notna()
            & self.data[self.prediction_column].notna()
        ].copy()

    def group_metrics(self) -> Dict[Hashable, Dict[str, Any]]:
        """
        Calculate representation and selection metrics for each group.
        """
        valid = self.valid_data()

        if valid.empty:
            return {}

        total_valid = len(valid)
        metrics: Dict[Hashable, Dict[str, Any]] = {}

        for group, group_data in valid.groupby(
            self.sensitive_attribute,
            sort=False,
        ):
            group_count = len(group_data)

            positive_count = int(
                (
                    group_data[self.prediction_column]
                    == self.positive_label
                ).sum()
            )

            selection_rate = (
                positive_count / group_count
                if group_count
                else 0.0
            )

            representation_rate = (
                group_count / total_valid
                if total_valid
                else 0.0
            )

            metrics[group] = {
                "count": int(group_count),
                "representation_rate": round(
                    float(representation_rate),
                    4,
                ),
                "positive_count": positive_count,
                "selection_rate": round(
                    float(selection_rate),
                    4,
                ),
            }

        return metrics

    def resolved_reference_group(
        self,
        metrics: Dict[Hashable, Dict[str, Any]],
    ) -> Hashable | None:
        """
        Determine the group used as the comparison reference.

        If a reference group was explicitly supplied, it is used.
        Otherwise the group with the highest selection rate is used.
        """
        if not metrics:
            return None

        if self.reference_group is not None:
            return self.reference_group

        return max(
            metrics,
            key=lambda group: metrics[group]["selection_rate"],
        )

    def comparative_metrics(
        self,
        metrics: Dict[Hashable, Dict[str, Any]],
        reference_group: Hashable | None,
    ) -> Dict[Hashable, Dict[str, Any]]:
        """
        Compare each group with the resolved reference group.
        """
        if reference_group is None:
            return {}

        reference_rate = metrics[reference_group]["selection_rate"]

        comparisons: Dict[Hashable, Dict[str, Any]] = {}

        for group, group_data in metrics.items():
            selection_rate = group_data["selection_rate"]

            rate_difference = selection_rate - reference_rate

            if reference_rate > 0:
                disparate_impact_ratio = (
                    selection_rate / reference_rate
                )
            elif selection_rate == 0:
                disparate_impact_ratio = 1.0
            else:
                disparate_impact_ratio = None

            requires_review = (
                disparate_impact_ratio is not None
                and disparate_impact_ratio
                < self.disparate_impact_threshold
            )

            comparisons[group] = {
                "selection_rate": selection_rate,
                "reference_selection_rate": reference_rate,
                "selection_rate_difference": round(
                    float(rate_difference),
                    4,
                ),
                "disparate_impact_ratio": (
                    round(
                        float(disparate_impact_ratio),
                        4,
                    )
                    if disparate_impact_ratio is not None
                    else None
                ),
                "requires_review": bool(requires_review),
            }

        return comparisons

    def missing_data_summary(self) -> Dict[str, int]:
        """Summarise missing values relevant to fairness analysis."""
        missing_sensitive = int(
            self.data[self.sensitive_attribute].isna().sum()
        )

        missing_predictions = int(
            self.data[self.prediction_column].isna().sum()
        )

        excluded_rows = int(
            (
                self.data[self.sensitive_attribute].isna()
                | self.data[self.prediction_column].isna()
            ).sum()
        )

        return {
            "missing_sensitive_attribute": missing_sensitive,
            "missing_predictions": missing_predictions,
            "excluded_rows": excluded_rows,
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the group fairness assessment.

        Returns
        -------
        dict
            Group metrics, comparative fairness indicators, missing-data
            information, and groups whose disparate-impact ratio falls
            below the configured review threshold.
        """
        metrics = self.group_metrics()

        reference_group = self.resolved_reference_group(metrics)

        comparisons = self.comparative_metrics(
            metrics,
            reference_group,
        )

        groups_requiring_review = [
            group
            for group, result in comparisons.items()
            if result["requires_review"]
        ]

        return {
            "sensitive_attribute": self.sensitive_attribute,
            "prediction_column": self.prediction_column,
            "positive_label": self.positive_label,
            "reference_group": reference_group,
            "disparate_impact_threshold": (
                self.disparate_impact_threshold
            ),
            "groups_analyzed": len(metrics),
            "group_metrics": metrics,
            "comparisons": comparisons,
            "groups_requiring_review": groups_requiring_review,
            "review_required": bool(groups_requiring_review),
            "missing_data": self.missing_data_summary(),
        }
