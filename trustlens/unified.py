
"""
Unified assessment interface for TrustLens AI.

This module coordinates the five TrustLens assessment dimensions:

- Data Quality
- Data Governance
- AI Readiness
- Responsible AI
- Operational Trust

The unified assessment preserves the detailed results produced by
each specialist assessor. It does not calculate an overall trust
score or certify that data or AI systems are safe, fair, reliable,
production-ready, or legally compliant.
"""

from typing import Any, Dict, Hashable, List, Optional

import pandas as pd

from .governance.assessor import DataGovernanceAssessor
from .operational_trust.assessor import OperationalTrustAssessor
from .quality.profiler import DataQualityProfiler
from .readiness.assessor import AIReadinessAssessor
from .responsible_ai.assessor import ResponsibleAIAssessor


class TrustLensAssessor:
    """
    Coordinate the five TrustLens AI assessment dimensions.

    Parameters
    ----------
    data:
        Non-empty pandas DataFrame to assess.
    target_column:
        Optional column containing observed outcomes.
    prediction_column:
        Optional column containing model predictions.
    sensitive_attributes:
        Optional list of attributes used for Responsible AI analysis.
    metadata:
        Optional dataset governance metadata.
    governance_controls:
        Optional dictionary describing governance controls.
    reference_data:
        Optional reference dataset for drift analysis.
    reproducibility_metadata:
        Optional reproducibility evidence.
    identifier_column:
        Optional row identifier for reproducibility analysis.
    monitoring_metadata:
        Optional operational monitoring evidence.
    positive_label:
        Positive outcome label for Responsible AI analysis.

    Additional specialist thresholds use the existing defaults
    of their respective assessors.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        prediction_column: Optional[str] = None,
        sensitive_attributes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        governance_controls: Optional[Dict[str, Any]] = None,
        reference_data: Optional[pd.DataFrame] = None,
        reproducibility_metadata: Optional[Dict[str, Any]] = None,
        identifier_column: Optional[str] = None,
        monitoring_metadata: Optional[Dict[str, Any]] = None,
        positive_label: Hashable = 1,
    ) -> None:
        """Validate and store unified assessment inputs."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if (
            sensitive_attributes is not None
            and not isinstance(sensitive_attributes, list)
        ):
            raise TypeError("sensitive_attributes must be a list or None")

        self.data = data.copy()
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.sensitive_attributes = (
            list(sensitive_attributes)
            if sensitive_attributes is not None
            else None
        )
        self.metadata = metadata
        self.governance_controls = governance_controls
        self.reference_data = (
            reference_data.copy()
            if isinstance(reference_data, pd.DataFrame)
            else reference_data
        )
        self.reproducibility_metadata = reproducibility_metadata
        self.identifier_column = identifier_column
        self.monitoring_metadata = monitoring_metadata
        self.positive_label = positive_label

    def assess(self) -> Dict[str, Any]:
        """
        Run all five TrustLens assessment dimensions.

        Returns
        -------
        dict
            Structured report containing assessment context,
            individual dimension results, analysis availability,
            and interpretation limitations.
        """
        quality = DataQualityProfiler(self.data).analyze()

        governance = DataGovernanceAssessor(
            self.data,
            metadata=self.metadata,
            governance_controls=self.governance_controls,
        ).assess()

        readiness = AIReadinessAssessor(
            self.data,
            target_column=self.target_column,
        ).assess()

        responsible_ai = ResponsibleAIAssessor(
            self.data,
            target_column=self.target_column,
            prediction_column=self.prediction_column,
            sensitive_attributes=self.sensitive_attributes,
            positive_label=self.positive_label,
        ).assess()

        operational_trust = OperationalTrustAssessor(
            self.data,
            reference_data=self.reference_data,
            reproducibility_metadata=self.reproducibility_metadata,
            identifier_column=self.identifier_column,
            monitoring_metadata=self.monitoring_metadata,
        ).assess()

        return {
            "framework": "TrustLens AI",
            "report_type": "unified_assessment",
            "assessment_context": {
                "dataset": {
                    "rows": int(self.data.shape[0]),
                    "columns": int(self.data.shape[1]),
                },
                "target_column": self.target_column,
                "prediction_column": self.prediction_column,
                "sensitive_attributes": (
                    list(self.sensitive_attributes or [])
                ),
                "reference_data_supplied": (
                    self.reference_data is not None
                ),
                "governance_metadata_supplied": (
                    self.metadata is not None
                ),
                "governance_controls_supplied": (
                    self.governance_controls is not None
                ),
                "reproducibility_metadata_supplied": (
                    self.reproducibility_metadata is not None
                ),
                "monitoring_metadata_supplied": (
                    self.monitoring_metadata is not None
                ),
            },
            "dimensions": {
                "data_quality": quality,
                "data_governance": governance,
                "ai_readiness": readiness,
                "responsible_ai": responsible_ai,
                "operational_trust": operational_trust,
            },
            "analysis_availability": {
                "governance": {
                    "metadata_completeness": (
                        governance["metadata_completeness"] is not None
                    ),
                    "governance_controls": (
                        governance["governance_controls"] is not None
                    ),
                },
                "ai_readiness": {
                    "class_imbalance": readiness[
                        "class_imbalance"
                    ].get("applicable", False),
                    "leakage_risk": readiness[
                        "leakage_risk"
                    ].get("applicable", False),
                },
                "responsible_ai": responsible_ai[
                    "analysis_availability"
                ],
                "operational_trust": operational_trust[
                    "analysis_availability"
                ],
            },
            "limitations": [
                (
                    "Assessment results describe available evidence "
                    "and indicators; they do not independently certify "
                    "trustworthiness, fairness, safety, or compliance."
                ),
                (
                    "Some analyses require optional metadata, model "
                    "predictions, sensitive attributes, or reference "
                    "datasets. Missing inputs limit assessment coverage."
                ),
                (
                    "Dimension-level findings must be interpreted "
                    "within the intended application, data context, "
                    "and applicable governance requirements."
                ),
            ],
        }

    def analyze(self) -> Dict[str, Any]:
        """Provide an alternative name for the unified assessment."""
        return self.assess()
