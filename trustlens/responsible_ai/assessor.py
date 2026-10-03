"""
Core Responsible AI assessor for TrustLens AI.

This module provides the central integration point for evaluating
responsible AI considerations associated with datasets and AI systems.

The assessor coordinates structural Responsible AI context with
specialised analysis components including group fairness, potential
bias indicators, prediction-performance fairness, and explainability
readiness.
"""

from typing import Any, Dict, Hashable, List

import pandas as pd

from .bias import BiasIndicatorAnalyzer
from .explainability import ExplainabilityAnalyzer
from .fairness import FairnessAnalyzer
from .performance_fairness import PerformanceFairnessAnalyzer


class ResponsibleAIAssessor:
    """
    Perform integrated Responsible AI assessment.

    The assessor stores dataset and modelling context and coordinates
    specialised Responsible AI analyses.

    Group fairness analysis evaluates model prediction disparities
    across configured sensitive attributes.

    Bias indicator analysis evaluates representation, group-size, and
    observed outcome disparities in the underlying data.

    Prediction-performance fairness analysis evaluates whether model
    performance and error metrics differ across configured sensitive
    attributes.

    Explainability-readiness analysis evaluates whether candidate
    explanatory features have structural characteristics suitable for
    meaningful downstream explanation analysis.

    These indicators identify conditions that may warrant review. They
    do not independently establish whether a dataset or model is fair,
    biased, discriminatory, explainable, ethical, or legally compliant.
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
        performance_gap_threshold: float = 0.10,
        minimum_group_size: int = 5,
        high_cardinality_threshold: int = 20,
        missingness_threshold: float = 0.50,
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
            Value treated as the positive outcome during fairness,
            observed-outcome, and predictive-performance analysis.
        disparate_impact_threshold:
            Ratio below which a prediction-rate comparison is flagged
            for fairness review.
        representation_threshold:
            Minimum proportion of valid records expected for each group
            during bias-indicator analysis.
        outcome_ratio_threshold:
            Ratio below which an observed outcome-rate comparison is
            flagged as a potential bias indicator.
        performance_gap_threshold:
            Maximum absolute difference between a group's predictive
            performance metric and the overall metric before a
            performance review indicator is raised.
        minimum_group_size:
            Minimum number of valid observations expected for each group
            during analyses that assess evidence sufficiency.
        high_cardinality_threshold:
            Number of unique values above which a categorical candidate
            explanatory feature is flagged as high cardinality.
        missingness_threshold:
            Proportion of missing values at or above which a candidate
            explanatory feature is flagged for review.

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

        self.performance_gap_threshold = float(
            performance_gap_threshold
        )

        self.minimum_group_size = minimum_group_size

        self.high_cardinality_threshold = (
            high_cardinality_threshold
        )

        self.missingness_threshold = float(
            missingness_threshold
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

    def explainability_candidate_features(self) -> List[str]:
        """
        Return candidate features available for explainability analysis.

        Target, prediction, and configured sensitive attributes are
        excluded from candidate explanatory features.
        """
        excluded = set(self.sensitive_attributes)

        if self.target_column is not None:
            excluded.add(self.target_column)

        if self.prediction_column is not None:
            excluded.add(self.prediction_column)

        return [
            column
            for column in self.data.columns
            if column not in excluded
        ]

    def analysis_availability(self) -> Dict[str, Any]:
        """
        Report which Responsible AI analyses have enough context.

        Availability indicates whether minimum structural inputs are
        present. It does not indicate that an analysis has established
        fairness, bias, discrimination, explainability, or compliance.
        """
        has_sensitive_attributes = self.sensitive_attribute_count > 0
        has_target = self.target_column is not None
        has_predictions = self.prediction_column is not None

        has_explainability_features = bool(
            self.explainability_candidate_features()
        )

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
                "available": has_explainability_features,
                "requires": [
                    "candidate_explanatory_features",
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

    def prediction_performance_fairness_analysis(
        self,
    ) -> Dict[str, Any]:
        """
        Run prediction-performance fairness analysis.

        Each configured sensitive attribute is analysed independently.

        The analysis compares group-level predictive performance with
        overall valid-dataset performance using classification metrics
        including accuracy, precision, recall, error rate,
        false-positive rate, and false-negative rate.

        Performance differences are descriptive review indicators and
        do not independently establish unfairness or discrimination.

        Returns a not-applicable result when the target column,
        prediction column, or sensitive attributes have not been
        supplied.
        """
        if self.target_column is None:
            return {
                "applicable": False,
                "reason": (
                    "Prediction-performance fairness analysis "
                    "requires a target column."
                ),
                "attributes": {},
                "review_required": False,
                "attributes_requiring_review": [],
            }

        if self.prediction_column is None:
            return {
                "applicable": False,
                "reason": (
                    "Prediction-performance fairness analysis "
                    "requires a prediction column."
                ),
                "attributes": {},
                "review_required": False,
                "attributes_requiring_review": [],
            }

        if not self.sensitive_attributes:
            return {
                "applicable": False,
                "reason": (
                    "Prediction-performance fairness analysis "
                    "requires at least one sensitive attribute."
                ),
                "attributes": {},
                "review_required": False,
                "attributes_requiring_review": [],
            }

        attribute_results: Dict[str, Any] = {}

        for attribute in self.sensitive_attributes:
            analyzer = PerformanceFairnessAnalyzer(
                self.data,
                sensitive_attribute=attribute,
                target_column=self.target_column,
                prediction_column=self.prediction_column,
                positive_label=self.positive_label,
                performance_gap_threshold=(
                    self.performance_gap_threshold
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
            "performance_gap_threshold": (
                self.performance_gap_threshold
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

    def explainability_analysis(self) -> Dict[str, Any]:
        """
        Run feature-level explainability-readiness analysis.

        The analysis can operate without model predictions because it
        evaluates whether suitable candidate explanatory features exist
        and whether their structure presents explanation-readiness
        concerns.

        Target, prediction, and configured sensitive attributes are
        excluded from candidate explanatory features.
        """
        analyzer = ExplainabilityAnalyzer(
            self.data,
            target_column=self.target_column,
            prediction_column=self.prediction_column,
            sensitive_attributes=self.sensitive_attributes,
            high_cardinality_threshold=(
                self.high_cardinality_threshold
            ),
            missingness_threshold=(
                self.missingness_threshold
            ),
        )

        result = analyzer.assess()

        return {
            "applicable": bool(
                result["candidate_feature_count"]
            ),
            **result,
        }

    def responsible_ai_summary(
        self,
        group_fairness: Dict[str, Any],
        bias_indicators: Dict[str, Any],
        performance_fairness: Dict[str, Any],
        explainability: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the integrated Responsible AI review summary.

        Review reasons are based on implemented TrustLens components.
        They represent indicators requiring further investigation rather
        than determinations of fairness, bias, explainability, ethics,
        or legal compliance.
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

        if (
            performance_fairness.get("applicable", False)
            and performance_fairness.get("review_required", False)
        ):
            review_reasons.append(
                "prediction_performance_fairness"
            )

        if explainability.get("review_required", False):
            review_reasons.append("explainability")

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
            "prediction_performance_fairness_applicable": (
                performance_fairness.get(
                    "applicable",
                    False,
                )
            ),
            "explainability_applicable": (
                explainability.get(
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
            "performance_attributes_requiring_review": (
                performance_fairness.get(
                    "attributes_requiring_review",
                    [],
                )
            ),
            "explainability_features_requiring_review": (
                explainability.get(
                    "review_summary",
                    {},
                ).get(
                    "features_requiring_review",
                    [],
                )
            ),
            "explanation_readiness_status": (
                explainability.get(
                    "explanation_readiness",
                    {},
                ).get(
                    "status",
                    "unknown",
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
            indicators, prediction-performance fairness, explainability
            readiness, and the integrated review summary.
        """
        group_fairness = self.group_fairness_analysis()
        bias_indicators = self.bias_indicator_analysis()

        performance_fairness = (
            self.prediction_performance_fairness_analysis()
        )

        explainability = self.explainability_analysis()

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
            "prediction_performance_fairness": (
                performance_fairness
            ),
            "explainability": explainability,
            "responsible_ai_summary": (
                self.responsible_ai_summary(
                    group_fairness,
                    bias_indicators,
                    performance_fairness,
                    explainability,
                )
            ),
        }
