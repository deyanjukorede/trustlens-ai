"""
Bias indicator analysis for TrustLens AI.

This module provides descriptive indicators for identifying potential
representation and observed-outcome disparities across groups in a
dataset.

The indicators are intended to support investigation and human review.
They do not independently establish that a dataset, organisation, or
AI system is biased, discriminatory, unethical, or legally non-compliant.
"""

from typing import Any, Dict, Hashable

import pandas as pd


class BiasIndicatorAnalyzer:
    """
    Analyse potential group-level bias indicators in a dataset.

    The analyzer examines representation, group size, and observed
    target outcomes across groups defined by a sensitive attribute.

    Parameters
    ----------
    data:
        Dataset containing the sensitive attribute and target.
    sensitive_attribute:
        Column used to define groups for analysis.
    target_column:
        Column containing observed outcomes.
    positive_label:
        Target value treated as the positive outcome.
    representation_threshold:
        Minimum proportion of valid records expected for each group
        before an under-representation indicator is raised.
    outcome_ratio_threshold:
        Ratio below which a group's positive outcome rate is flagged
        relative to the group with the highest observed outcome rate.
    minimum_group_size:
        Minimum number of valid observations expected for a group
        before a small-group indicator is raised.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sensitive_attribute: str,
        target_column: str,
        positive_label: Hashable = 1,
        representation_threshold: float = 0.10,
        outcome_ratio_threshold: float = 0.80,
        minimum_group_size: int = 5,
    ) -> None:
        """Initialise the bias indicator analyzer."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if sensitive_attribute not in data.columns:
            raise ValueError(
                "sensitive_attribute "
                f"'{sensitive_attribute}' does not exist in the dataset"
            )

        if target_column not in data.columns:
            raise ValueError(
                f"target_column '{target_column}' does not exist in the dataset"
            )

        if not 0 < representation_threshold <= 1:
            raise ValueError(
                "representation_threshold must be greater than 0 "
                "and less than or equal to 1"
            )

        if not 0 < outcome_ratio_threshold <= 1:
            raise ValueError(
                "outcome_ratio_threshold must be greater than 0 "
                "and less than or equal to 1"
            )

        if (
            not isinstance(minimum_group_size, int)
            or isinstance(minimum_group_size, bool)
            or minimum_group_size < 1
        ):
            raise ValueError(
                "minimum_group_size must be a positive integer"
            )

        self.data = data
        self.sensitive_attribute = sensitive_attribute
        self.target_column = target_column
        self.positive_label = positive_label
        self.representation_threshold = float(
            representation_threshold
        )
        self.outcome_ratio_threshold = float(
            outcome_ratio_threshold
        )
        self.minimum_group_size = minimum_group_size

    def valid_data(self) -> pd.DataFrame:
        """
        Return rows with non-missing group and target values.

        Missing sensitive attributes or target outcomes are excluded
        from group calculations and reported separately.
        """
        return self.data[
            self.data[self.sensitive_attribute].notna()
            & self.data[self.target_column].notna()
        ].copy()

    def group_metrics(self) -> Dict[Hashable, Dict[str, Any]]:
        """
        Calculate representation and observed outcome metrics by group.
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
                    group_data[self.target_column]
                    == self.positive_label
                ).sum()
            )

            representation_rate = (
                group_count / total_valid
                if total_valid
                else 0.0
            )

            positive_outcome_rate = (
                positive_count / group_count
                if group_count
                else 0.0
            )

            metrics[group] = {
                "count": int(group_count),
                "representation_rate": round(
                    float(representation_rate),
                    4,
                ),
                "positive_count": positive_count,
                "positive_outcome_rate": round(
                    float(positive_outcome_rate),
                    4,
                ),
                "underrepresented": bool(
                    representation_rate
                    < self.representation_threshold
                ),
                "small_group": bool(
                    group_count < self.minimum_group_size
                ),
            }

        return metrics

    def reference_group(
        self,
        metrics: Dict[Hashable, Dict[str, Any]],
    ) -> Hashable | None:
        """
        Return the group with the highest observed positive outcome rate.
        """
        if not metrics:
            return None

        return max(
            metrics,
            key=lambda group: metrics[group][
                "positive_outcome_rate"
            ],
        )

    def outcome_comparisons(
        self,
        metrics: Dict[Hashable, Dict[str, Any]],
        reference_group: Hashable | None,
    ) -> Dict[Hashable, Dict[str, Any]]:
        """
        Compare observed positive outcome rates across groups.
        """
        if reference_group is None:
            return {}

        reference_rate = metrics[reference_group][
            "positive_outcome_rate"
        ]

        comparisons: Dict[Hashable, Dict[str, Any]] = {}

        for group, group_data in metrics.items():
            group_rate = group_data["positive_outcome_rate"]

            rate_difference = group_rate - reference_rate

            if reference_rate > 0:
                outcome_rate_ratio = group_rate / reference_rate
            elif group_rate == 0:
                outcome_rate_ratio = 1.0
            else:
                outcome_rate_ratio = None

            outcome_disparity = (
                outcome_rate_ratio is not None
                and outcome_rate_ratio
                < self.outcome_ratio_threshold
            )

            comparisons[group] = {
                "positive_outcome_rate": group_rate,
                "reference_outcome_rate": reference_rate,
                "outcome_rate_difference": round(
                    float(rate_difference),
                    4,
                ),
                "outcome_rate_ratio": (
                    round(
                        float(outcome_rate_ratio),
                        4,
                    )
                    if outcome_rate_ratio is not None
                    else None
                ),
                "outcome_disparity": bool(
                    outcome_disparity
                ),
            }

        return comparisons

    def indicator_summary(
        self,
        metrics: Dict[Hashable, Dict[str, Any]],
        comparisons: Dict[Hashable, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Summarise representation, group-size, and outcome indicators.
        """
        underrepresented_groups = [
            group
            for group, result in metrics.items()
            if result["underrepresented"]
        ]

        small_groups = [
            group
            for group, result in metrics.items()
            if result["small_group"]
        ]

        outcome_disparity_groups = [
            group
            for group, result in comparisons.items()
            if result["outcome_disparity"]
        ]

        indicators: Dict[Hashable, list[str]] = {}

        for group in metrics:
            group_indicators: list[str] = []

            if metrics[group]["underrepresented"]:
                group_indicators.append(
                    "underrepresentation"
                )

            if metrics[group]["small_group"]:
                group_indicators.append(
                    "small_group"
                )

            if comparisons.get(
                group,
                {},
            ).get("outcome_disparity", False):
                group_indicators.append(
                    "outcome_disparity"
                )

            if group_indicators:
                indicators[group] = group_indicators

        return {
            "underrepresented_groups": underrepresented_groups,
            "small_groups": small_groups,
            "outcome_disparity_groups": outcome_disparity_groups,
            "groups_with_indicators": list(indicators.keys()),
            "indicators_by_group": indicators,
            "review_required": bool(indicators),
        }

    def missing_data_summary(self) -> Dict[str, int]:
        """Summarise missing values relevant to bias analysis."""
        missing_sensitive = int(
            self.data[self.sensitive_attribute].isna().sum()
        )

        missing_target = int(
            self.data[self.target_column].isna().sum()
        )

        excluded_rows = int(
            (
                self.data[self.sensitive_attribute].isna()
                | self.data[self.target_column].isna()
            ).sum()
        )

        return {
            "missing_sensitive_attribute": missing_sensitive,
            "missing_target": missing_target,
            "excluded_rows": excluded_rows,
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the bias indicator assessment.

        Returns
        -------
        dict
            Representation metrics, observed outcome comparisons,
            potential bias indicators, and missing-data information.
        """
        metrics = self.group_metrics()

        reference_group = self.reference_group(metrics)

        comparisons = self.outcome_comparisons(
            metrics,
            reference_group,
        )

        indicators = self.indicator_summary(
            metrics,
            comparisons,
        )

        return {
            "sensitive_attribute": self.sensitive_attribute,
            "target_column": self.target_column,
            "positive_label": self.positive_label,
            "representation_threshold": (
                self.representation_threshold
            ),
            "outcome_ratio_threshold": (
                self.outcome_ratio_threshold
            ),
            "minimum_group_size": self.minimum_group_size,
            "groups_analyzed": len(metrics),
            "reference_group": reference_group,
            "group_metrics": metrics,
            "outcome_comparisons": comparisons,
            "indicators": indicators,
            "review_required": indicators["review_required"],
            "missing_data": self.missing_data_summary(),
        }
