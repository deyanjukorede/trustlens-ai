"""
Core Responsible AI assessor for TrustLens AI.

This module provides the central integration point for evaluating
responsible AI considerations associated with datasets and AI systems.

The assessor coordinates structural Responsible AI context with
specialised analysis components including group fairness and potential
bias indicators. Additional performance-fairness and explainability
capabilities can be integrated as Phase 5 develops.
"""

from typing import Any, Dict, Hashable, List

import pandas as pd

from .bias import BiasIndicatorAnalyzer
from .fairness import FairnessAnalyzer


class ResponsibleAIAssessor:
    """
    Perform integrated Responsible AI assessment.

    The assessor stores dataset and modelling context and coordinates
    specialised Responsible AI analyses.

    Group fairness analysis evaluates model prediction disparities
    across configured sensitive attributes.

    Bias indicator analysis evaluates representation, group-size, and
    observed outcome disparities in the underlying data.

    These indicators identify conditions that may warrant review. They
    do not independently establish whether a dataset or model is fair,
    biased, discriminatory, ethical, or legally compliant.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
        prediction_column: str | None = None,
        sensitive_attributes: List[str] | None = None,
        positive_label: Hashable = 1,
        disparate_impact_threshold: float = 0.80,
        representation_threshold: float = 0.10,
        outcome_ratio_threshold: float = 0.80,
        minimum_group_size: int = 5,
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
            Optional list of attributes across which Responsible AI
            analyses may be performed.
        positive_label:
            Value treated as the positive outcome during fairness and
            observed-outcome analysis.
        disparate_impact_threshold:
            Ratio below which a prediction-rate comparison is flagged
            for fairness review.
        representation_threshold:
            Minimum proportion of valid records expected for each group
            during bias-indicator analysis.
        outcome_ratio_threshold:
            Ratio below which an observed outcome-rate comparison is
            flagged as a potential bias indicator.
        minimum_group_size:
            Minimum number of valid observations expected for each group.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame or sensitive_attributes
            is not a list when supplied.
        ValueError
            If the dataset is empty, supplied columns do not exist, or
            configurable thresholds are invalid.
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

        self.representation_threshold = float(
            representation_threshold
        )

        self.outcome_ratio_threshold = float(
            outcome_ratio_threshold
        )

        self.minimum_group_size = minimum_group_size

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
        Run prediction-based group fairness analysis.

        Each configured sensitive attribute is analysed independently.

        Returns a not-applicable result when predictions or sensitive
        attributes have not been supplied.
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

    def bias_indicator_analysis(self) -> Dict[str, Any]:
        """
        Run dataset and observed-outcome bias indicator analysis.

        Each configured sensitive attribute is analysed independently.

        Bias indicators describe representation, group-size, and
        observed outcome disparities that may warrant investigation.
        They do not establish that bias or discrimination exists.

        Returns a not-applicable result when the target column or
        sensitive attributes have not been supplied.
        """
        if self.target_column is None:
            return {
                "applicable": False,
                "reason": (
                    "Bias indicator analysis requires a target column."
                ),
                "attributes": {},
                "review_required": False,
                "attributes_requiring_review": [],
            }

        if not self.sensitive_attributes:
            return {
                "applicable": False,
                "reason": (
                    "Bias indicator analysis requires at least one "
                    "sensitive attribute."
                ),
                "attributes": {},
                "review_required": False,
                "attributes_requiring_review": [],
            }

        attribute_results: Dict[str, Any] = {}

        for attribute in self.sensitive_attributes:
            analyzer = BiasIndicatorAnalyzer(
                self.data,
                sensitive_attribute=attribute,
                target_column=self.target_column,
                positive_label=self.positive_label,
                representation_threshold=(
                    self.representation_threshold
                ),
                outcome_ratio_threshold=(
                    self.outcome_ratio_threshold
                ),
                minimum_group_size=self.minimum_group_size,
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
            "representation_threshold": (
                self.representation_threshold
            ),
            "outcome_ratio_threshold": (
                self.outcome_ratio_threshold
            ),
            "minimum_group_size": self.minimum_group_size,
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
        bias_indicators: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the integrated Responsible AI review summary.

        Review reasons are based only on implemented TrustLens
        components. Future Responsible AI components can contribute
        additional indicators as Phase 5 develops.
        """
        review_reasons: List[str] = []

        if (
            group_fairness.get("applicable", False)
            and group_fairness.get("review_required", False)
        ):
            review_reasons.append("group_fairness")

        if (
            bias_indicators.get("applicable", False)
            and bias_indicators.get("review_required", False)
        ):
            review_reasons.append("bias_indicators")

        status = (
            "review"
            if review_reasons
            else "no_review_indicators"
        )

        return {
            "status": status,
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "group_fairness_applicable": (
                group_fairness.get(
                    "applicable",
                    False,
                )
            ),
            "bias_indicators_applicable": (
                bias_indicators.get(
                    "applicable",
                    False,
                )
            ),
            "fairness_attributes_requiring_review": (
                group_fairness.get(
                    "attributes_requiring_review",
                    [],
                )
            ),
            "bias_attributes_requiring_review": (
                bias_indicators.get(
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
            availability information, group fairness analysis, bias
            indicators, and the integrated review summary.
        """
        group_fairness = self.group_fairness_analysis()
        bias_indicators = self.bias_indicator_analysis()

        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "target_column": self.target_column,
            "prediction_column": self.prediction_column,
            "sensitive_attributes": list(
                self.sensitive_attributes
            ),
            "assessment_context": self.assessment_context(),
            "sensitive_attribute_summary": (
                self.sensitive_attribute_summary()
            ),
            "analysis_availability": (
                self.analysis_availability()
            ),
            "group_fairness": group_fairness,
            "bias_indicators": bias_indicators,
            "responsible_ai_summary": (
                self.responsible_ai_summary(
                    group_fairness,
                    bias_indicators,
                )
            ),
        }
