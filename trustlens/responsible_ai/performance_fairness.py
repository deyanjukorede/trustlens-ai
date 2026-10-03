"""
Prediction-performance fairness analysis for TrustLens AI.

This module provides descriptive indicators for comparing predictive
performance across groups defined by a sensitive attribute.

The analyzer calculates group-level classification metrics including
accuracy, precision, recall, error rate, false-positive rate, and
false-negative rate. It then identifies measurable performance gaps
that may warrant further human review.

These indicators do not independently establish that a model is fair,
unfair, discriminatory, ethical, or legally compliant.
"""

from typing import Any, Dict, Hashable

import pandas as pd


class PerformanceFairnessAnalyzer:
    """
    Analyse predictive performance across groups.

    The analyzer compares binary-classification performance for groups
    defined by a sensitive attribute.

    Parameters
    ----------
    data:
        Dataset containing sensitive attribute, target, and prediction.
    sensitive_attribute:
        Column used to define groups for comparison.
    target_column:
        Column containing observed outcomes.
    prediction_column:
        Column containing model predictions.
    positive_label:
        Value treated as the positive class.
    performance_gap_threshold:
        Maximum absolute difference tolerated between a group's metric
        and the reference metric before a review indicator is raised.
        The threshold is a configurable screening rule rather than a
        universal definition of performance fairness.
    minimum_group_size:
        Minimum number of valid observations expected for a group.
        Groups below this size are reported as having limited evidence.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sensitive_attribute: str,
        target_column: str,
        prediction_column: str,
        positive_label: Hashable = 1,
        performance_gap_threshold: float = 0.10,
        minimum_group_size: int = 5,
    ) -> None:
        """Initialise the performance fairness analyzer."""
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

        if prediction_column not in data.columns:
            raise ValueError(
                "prediction_column "
                f"'{prediction_column}' does not exist in the dataset"
            )

        if not 0 <= performance_gap_threshold <= 1:
            raise ValueError(
                "performance_gap_threshold must be between 0 and 1"
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
        self.prediction_column = prediction_column
        self.positive_label = positive_label
        self.performance_gap_threshold = float(
            performance_gap_threshold
        )
        self.minimum_group_size = minimum_group_size

    def valid_data(self) -> pd.DataFrame:
        """
        Return rows containing group, target, and prediction values.

        Rows with missing sensitive attributes, observed outcomes, or
        predictions are excluded from metric calculations and reported
        separately.
        """
        return self.data[
            self.data[self.sensitive_attribute].notna()
            & self.data[self.target_column].notna()
            & self.data[self.prediction_column].notna()
        ].copy()

    def confusion_counts(
        self,
        group_data: pd.DataFrame,
    ) -> Dict[str, int]:
        """
        Calculate binary confusion-matrix counts for a group.

        Any value other than positive_label is treated as the negative
        class for this binary screening analysis.
        """
        actual_positive = (
            group_data[self.target_column]
            == self.positive_label
        )

        predicted_positive = (
            group_data[self.prediction_column]
            == self.positive_label
        )

        true_positive = int(
            (actual_positive & predicted_positive).sum()
        )

        true_negative = int(
            (~actual_positive & ~predicted_positive).sum()
        )

        false_positive = int(
            (~actual_positive & predicted_positive).sum()
        )

        false_negative = int(
            (actual_positive & ~predicted_positive).sum()
        )

        return {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    def calculate_metrics(
        self,
        counts: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Calculate classification metrics from confusion counts.
        """
        true_positive = counts["true_positive"]
        true_negative = counts["true_negative"]
        false_positive = counts["false_positive"]
        false_negative = counts["false_negative"]

        total = (
            true_positive
            + true_negative
            + false_positive
            + false_negative
        )

        predicted_positive = true_positive + false_positive
        actual_positive = true_positive + false_negative
        actual_negative = true_negative + false_positive

        accuracy = (
            (true_positive + true_negative) / total
            if total
            else 0.0
        )

        error_rate = (
            (false_positive + false_negative) / total
            if total
            else 0.0
        )

        precision = (
            true_positive / predicted_positive
            if predicted_positive
            else 0.0
        )

        recall = (
            true_positive / actual_positive
            if actual_positive
            else 0.0
        )

        false_positive_rate = (
            false_positive / actual_negative
            if actual_negative
            else 0.0
        )

        false_negative_rate = (
            false_negative / actual_positive
            if actual_positive
            else 0.0
        )

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "error_rate": round(float(error_rate), 4),
            "false_positive_rate": round(
                float(false_positive_rate),
                4,
            ),
            "false_negative_rate": round(
                float(false_negative_rate),
                4,
            ),
        }

    def group_metrics(self) -> Dict[Hashable, Dict[str, Any]]:
        """
        Calculate predictive performance metrics for each group.
        """
        valid = self.valid_data()

        if valid.empty:
            return {}

        metrics: Dict[Hashable, Dict[str, Any]] = {}

        for group, group_data in valid.groupby(
            self.sensitive_attribute,
            sort=False,
        ):
            counts = self.confusion_counts(group_data)
            performance = self.calculate_metrics(counts)

            metrics[group] = {
                "count": int(len(group_data)),
                "limited_evidence": bool(
                    len(group_data) < self.minimum_group_size
                ),
                "confusion_counts": counts,
                **performance,
            }

        return metrics

    def overall_metrics(self) -> Dict[str, Any]:
        """
        Calculate predictive performance across all valid observations.

        Overall dataset performance is used as a neutral comparison
        baseline for identifying group-level metric gaps.
        """
        valid = self.valid_data()

        if valid.empty:
            return {
                "count": 0,
                "confusion_counts": {
                    "true_positive": 0,
                    "true_negative": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                },
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "error_rate": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
            }

        counts = self.confusion_counts(valid)
        performance = self.calculate_metrics(counts)

        return {
            "count": int(len(valid)),
            "confusion_counts": counts,
            **performance,
        }

    def performance_comparisons(
        self,
        metrics: Dict[Hashable, Dict[str, Any]],
        overall: Dict[str, Any],
    ) -> Dict[Hashable, Dict[str, Any]]:
        """
        Compare group performance metrics with overall performance.

        Absolute metric gaps are used so that both higher and lower
        group values can be surfaced for review.
        """
        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "error_rate",
            "false_positive_rate",
            "false_negative_rate",
        ]

        comparisons: Dict[Hashable, Dict[str, Any]] = {}

        for group, group_result in metrics.items():
            metric_gaps: Dict[str, float] = {}
            flagged_metrics = []

            for metric in metric_names:
                group_value = group_result[metric]
                overall_value = overall[metric]

                gap = abs(group_value - overall_value)

                metric_gaps[metric] = round(
                    float(gap),
                    4,
                )

                if gap > self.performance_gap_threshold:
                    flagged_metrics.append(metric)

            comparisons[group] = {
                "metric_gaps": metric_gaps,
                "flagged_metrics": flagged_metrics,
                "performance_gap_detected": bool(
                    flagged_metrics
                ),
                "limited_evidence": group_result[
                    "limited_evidence"
                ],
                "requires_review": bool(
                    flagged_metrics
                    or group_result["limited_evidence"]
                ),
            }

        return comparisons

    def review_summary(
        self,
        comparisons: Dict[Hashable, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Summarise performance and evidence-sufficiency indicators.
        """
        performance_gap_groups = [
            group
            for group, result in comparisons.items()
            if result["performance_gap_detected"]
        ]

        limited_evidence_groups = [
            group
            for group, result in comparisons.items()
            if result["limited_evidence"]
        ]

        groups_requiring_review = [
            group
            for group, result in comparisons.items()
            if result["requires_review"]
        ]

        return {
            "performance_gap_groups": performance_gap_groups,
            "limited_evidence_groups": limited_evidence_groups,
            "groups_requiring_review": groups_requiring_review,
            "review_required": bool(groups_requiring_review),
        }

    def missing_data_summary(self) -> Dict[str, int]:
        """
        Summarise missing values relevant to performance analysis.
        """
        missing_sensitive = int(
            self.data[self.sensitive_attribute].isna().sum()
        )

        missing_target = int(
            self.data[self.target_column].isna().sum()
        )

        missing_predictions = int(
            self.data[self.prediction_column].isna().sum()
        )

        excluded_rows = int(
            (
                self.data[self.sensitive_attribute].isna()
                | self.data[self.target_column].isna()
                | self.data[self.prediction_column].isna()
            ).sum()
        )

        return {
            "missing_sensitive_attribute": missing_sensitive,
            "missing_target": missing_target,
            "missing_predictions": missing_predictions,
            "excluded_rows": excluded_rows,
        }

    def assess(self) -> Dict[str, Any]:
        """
        Run the prediction-performance fairness assessment.

        Returns
        -------
        dict
            Group performance metrics, overall performance, metric-gap
            comparisons, evidence-sufficiency indicators, review
            summary, and missing-data information.
        """
        metrics = self.group_metrics()
        overall = self.overall_metrics()

        comparisons = self.performance_comparisons(
            metrics,
            overall,
        )

        review = self.review_summary(comparisons)

        return {
            "sensitive_attribute": self.sensitive_attribute,
            "target_column": self.target_column,
            "prediction_column": self.prediction_column,
            "positive_label": self.positive_label,
            "performance_gap_threshold": (
                self.performance_gap_threshold
            ),
            "minimum_group_size": self.minimum_group_size,
            "groups_analyzed": len(metrics),
            "overall_metrics": overall,
            "group_metrics": metrics,
            "comparisons": comparisons,
            "review_summary": review,
            "review_required": review["review_required"],
            "missing_data": self.missing_data_summary(),
        }
