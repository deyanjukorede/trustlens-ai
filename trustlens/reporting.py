
"""
Cross-dimension reporting for TrustLens AI.

This module consolidates the outputs of the five specialist
assessment dimensions into a transparent review summary.

It distinguishes identified review indicators from unavailable
analyses and does not calculate an overall numerical trust score.

Results support human investigation and are not certifications
of fairness, safety, reliability, readiness, or compliance.
"""

from typing import Any, Dict, List


class UnifiedReportBuilder:
    """
    Build a consolidated report from five TrustLens dimensions.

    The builder expects the structured results produced by
    TrustLensAssessor and preserves the underlying specialist
    assessments without modifying their findings.
    """

    DIMENSIONS = (
        "data_quality",
        "data_governance",
        "ai_readiness",
        "responsible_ai",
        "operational_trust",
    )

    def __init__(self, assessment: Dict[str, Any]) -> None:
        """Validate the unified assessment structure."""
        if not isinstance(assessment, dict):
            raise TypeError("assessment must be a dictionary")

        dimensions = assessment.get("dimensions")

        if not isinstance(dimensions, dict):
            raise ValueError(
                "assessment must contain a dimensions dictionary"
            )

        missing = [
            name
            for name in self.DIMENSIONS
            if name not in dimensions
        ]

        if missing:
            raise ValueError(
                "assessment is missing dimensions: "
                + ", ".join(missing)
            )

        for name in self.DIMENSIONS:
            if not isinstance(dimensions[name], dict):
                raise TypeError(
                    f"dimension '{name}' must be a dictionary"
                )

        self.assessment = assessment
        self.dimensions = dimensions

    @staticmethod
    def _add_review(
        reviews: List[Dict[str, Any]],
        dimension: str,
        indicator: str,
        detail: str,
    ) -> None:
        """Append an evidence-based human-review indicator."""
        reviews.append(
            {
                "dimension": dimension,
                "indicator": indicator,
                "detail": detail,
            }
        )

    @staticmethod
    def _add_unavailable(
        unavailable: List[Dict[str, str]],
        dimension: str,
        analysis: str,
        reason: str,
    ) -> None:
        """Record an analysis that could not be performed."""
        unavailable.append(
            {
                "dimension": dimension,
                "analysis": analysis,
                "reason": reason,
            }
        )

    def _quality_findings(
        self,
        reviews: List[Dict[str, Any]],
    ) -> None:
        """Surface observed quality issues without new thresholds."""
        quality = self.dimensions["data_quality"]

        missing = quality.get("missing_values", {})
        duplicates = quality.get("duplicates", {})

        if missing.get("total_missing", 0) > 0:
            self._add_review(
                reviews,
                "data_quality",
                "missing_values",
                "The dataset contains missing values.",
            )

        if duplicates.get("duplicate_count", 0) > 0:
            self._add_review(
                reviews,
                "data_quality",
                "duplicate_rows",
                "The dataset contains duplicate rows.",
            )

    def _governance_findings(
        self,
        reviews: List[Dict[str, Any]],
        unavailable: List[Dict[str, str]],
    ) -> None:
        """Preserve governance evidence availability."""
        governance = self.dimensions["data_governance"]

        for analysis, reason in (
            (
                "metadata_completeness",
                "Dataset governance metadata was not supplied.",
            ),
            (
                "governance_controls",
                "Governance-control evidence was not supplied.",
            ),
        ):
            if governance.get(analysis) is None:
                self._add_unavailable(
                    unavailable,
                    "data_governance",
                    analysis,
                    reason,
                )

        # Preserve detailed privacy and governance findings
        # in the original specialist assessment. Do not infer
        # review decisions from undocumented risk labels.

    def _readiness_findings(
        self,
        reviews: List[Dict[str, Any]],
        unavailable: List[Dict[str, str]],
    ) -> None:
        """Consolidate existing AI-readiness review reasons."""
        readiness = self.dimensions["ai_readiness"]

        summary = readiness.get("readiness_summary", {})

        for reason in summary.get("review_reasons", []):
            self._add_review(
                reviews,
                "ai_readiness",
                str(reason),
                "The AI Readiness assessor identified a "
                "condition requiring further review.",
            )

        for analysis in ("class_imbalance", "leakage_risk"):
            result = readiness.get(analysis, {})

            if result.get("applicable") is False:
                self._add_unavailable(
                    unavailable,
                    "ai_readiness",
                    analysis,
                    str(
                        result.get(
                            "reason",
                            "Required modelling context was not supplied.",
                        )
                    ),
                )

    def _responsible_ai_findings(
        self,
        reviews: List[Dict[str, Any]],
        unavailable: List[Dict[str, str]],
    ) -> None:
        """Consolidate Responsible AI review and availability."""
        responsible = self.dimensions["responsible_ai"]

        summary = responsible.get("responsible_ai_summary", {})

        for reason in summary.get("review_reasons", []):
            self._add_review(
                reviews,
                "responsible_ai",
                str(reason),
                "The Responsible AI assessor identified an "
                "indicator requiring human investigation.",
            )

        for analysis in (
            "group_fairness",
            "bias_indicators",
            "prediction_performance_fairness",
            "explainability",
        ):
            result = responsible.get(analysis, {})

            if result.get("applicable") is False:
                self._add_unavailable(
                    unavailable,
                    "responsible_ai",
                    analysis,
                    str(
                        result.get(
                            "reason",
                            "Required assessment context is unavailable.",
                        )
                    ),
                )

    def _operational_findings(
        self,
        reviews: List[Dict[str, Any]],
        unavailable: List[Dict[str, str]],
    ) -> None:
        """Consolidate Operational Trust review indicators."""
        operational = self.dimensions["operational_trust"]

        summary = operational.get("operational_trust_summary", {})

        for reason in summary.get("review_reasons", []):
            self._add_review(
                reviews,
                "operational_trust",
                str(reason),
                "The Operational Trust assessor identified "
                "a condition requiring human review.",
            )

        if operational.get("data_drift") is None:
            self._add_unavailable(
                unavailable,
                "operational_trust",
                "data_drift",
                "Reference data was not supplied for drift analysis.",
            )

    def build(self) -> Dict[str, Any]:
        """
        Build the consolidated cross-dimension report.

        Returns
        -------
        dict
            Review indicators, unavailable analyses, dimension
            coverage, and interpretation limitations.
        """
        reviews: List[Dict[str, Any]] = []
        unavailable: List[Dict[str, str]] = []

        self._quality_findings(reviews)
        self._governance_findings(reviews, unavailable)
        self._readiness_findings(reviews, unavailable)
        self._responsible_ai_findings(reviews, unavailable)
        self._operational_findings(reviews, unavailable)

        dimensions_requiring_review = [
            dimension
            for dimension in self.DIMENSIONS
            if any(
                item["dimension"] == dimension
                for item in reviews
            )
        ]

        dimensions_with_unavailable_analyses = [
            dimension
            for dimension in self.DIMENSIONS
            if any(
                item["dimension"] == dimension
                for item in unavailable
            )
        ]

        return {
            "report_type": "cross_dimension_summary",
            "dimensions_assessed": list(self.DIMENSIONS),
            "dimension_count": len(self.DIMENSIONS),
            "review_required": bool(reviews),
            "review_indicator_count": len(reviews),
            "review_indicators": reviews,
            "dimensions_requiring_review": (
                dimensions_requiring_review
            ),
            "unavailable_analysis_count": len(unavailable),
            "unavailable_analyses": unavailable,
            "dimensions_with_unavailable_analyses": (
                dimensions_with_unavailable_analyses
            ),
            "interpretation": (
                "One or more assessment dimensions contain "
                "indicators requiring human review."
                if reviews
                else (
                    "No review indicators were identified by the "
                    "cross-dimension reporting rules. This does not "
                    "establish that the dataset or AI system is "
                    "trustworthy, safe, fair, or production-ready."
                )
            ),
            "limitations": list(
                self.assessment.get("limitations", [])
            ),
        }
