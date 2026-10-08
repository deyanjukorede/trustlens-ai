"""
Monitoring readiness analysis for TrustLens AI.

This module evaluates evidence that can support ongoing monitoring of
data and AI workflows in operational environments.

Monitoring readiness cannot establish that a system is effectively
monitored, reliable, safe, trustworthy, or compliant. The analyzer
therefore reports the monitoring evidence supplied, identifies missing
evidence, and surfaces indicators that may require human review.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


class MonitoringReadinessAnalyzer:
    """
    Assess evidence supporting operational monitoring readiness.

    Parameters
    ----------
    data:
        Current dataset being assessed.
    metadata:
        Optional dictionary describing monitoring evidence.

        Recognised fields are:

        - ``monitoring_metrics``
        - ``alerting_rules``
        - ``monitoring_owner``
        - ``review_cadence``
        - ``logging_enabled``
        - ``incident_process``
        - ``reassessment_triggers``
        - ``monitoring_history``

        Additional metadata fields are preserved but do not currently
        affect monitoring readiness indicators.
    """

    REQUIRED_MONITORING_FIELDS = (
        "monitoring_metrics",
        "alerting_rules",
        "monitoring_owner",
        "review_cadence",
        "logging_enabled",
        "incident_process",
        "reassessment_triggers",
        "monitoring_history",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the monitoring readiness analyzer."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                "metadata must be a dictionary when supplied"
            )

        self.data = data.copy()
        self.metadata = dict(metadata) if metadata is not None else {}

        self.row_count = int(len(self.data))
        self.column_count = int(len(self.data.columns))

    @staticmethod
    def _has_evidence(value: Any) -> bool:
        """
        Return whether a metadata value contains usable evidence.

        Boolean values are treated explicitly. ``True`` represents
        supplied affirmative evidence, while ``False`` represents an
        explicitly supplied negative control state.
        """
        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)

        if isinstance(value, bool):
            return True

        return True

    def monitoring_evidence(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate recognised monitoring evidence fields."""
        evidence: Dict[str, Dict[str, Any]] = {}

        for field in self.REQUIRED_MONITORING_FIELDS:
            value = self.metadata.get(field)

            evidence[field] = {
                "provided": self._has_evidence(value),
                "value": value,
            }

        return evidence

    def missing_monitoring_fields(self) -> List[str]:
        """Return recognised monitoring fields without evidence."""
        evidence = self.monitoring_evidence()

        return [
            field
            for field in self.REQUIRED_MONITORING_FIELDS
            if not evidence[field]["provided"]
        ]

    def additional_metadata_fields(self) -> List[str]:
        """Return metadata fields outside the recognised monitoring set."""
        return [
            field
            for field in self.metadata
            if field not in self.REQUIRED_MONITORING_FIELDS
        ]

    def evidence_coverage(self) -> Dict[str, Any]:
        """Summarise coverage of recognised monitoring evidence."""
        evidence = self.monitoring_evidence()

        provided_count = sum(
            1
            for result in evidence.values()
            if result["provided"]
        )

        total_fields = len(self.REQUIRED_MONITORING_FIELDS)

        coverage_rate = round(
            provided_count / total_fields,
            4,
        )

        return {
            "recognised_fields": total_fields,
            "provided_fields": provided_count,
            "missing_fields": total_fields - provided_count,
            "coverage_rate": coverage_rate,
        }

    def logging_analysis(self) -> Dict[str, Any]:
        """
        Evaluate explicitly supplied logging configuration evidence.

        Absence of the field is distinguished from an explicit False
        value so that missing evidence and a disabled control are not
        treated as the same condition.
        """
        supplied = "logging_enabled" in self.metadata

        if not supplied:
            return {
                "provided": False,
                "enabled": None,
                "requires_review": False,
            }

        value = self.metadata.get("logging_enabled")

        if not isinstance(value, bool):
            return {
                "provided": self._has_evidence(value),
                "enabled": None,
                "requires_review": True,
            }

        return {
            "provided": True,
            "enabled": value,
            "requires_review": value is False,
        }

    def monitoring_history_analysis(self) -> Dict[str, Any]:
        """Evaluate whether monitoring history evidence is available."""
        value = self.metadata.get("monitoring_history")
        provided = self._has_evidence(value)

        return {
            "provided": provided,
            "value": value,
            "requires_review": False,
        }

    def assess(self) -> Dict[str, Any]:
        """Return the complete monitoring readiness assessment."""
        evidence = self.monitoring_evidence()
        missing_fields = self.missing_monitoring_fields()
        additional_fields = self.additional_metadata_fields()
        coverage = self.evidence_coverage()
        logging = self.logging_analysis()
        monitoring_history = self.monitoring_history_analysis()

        review_reasons: List[str] = []

        if missing_fields:
            review_reasons.append(
                "missing_monitoring_evidence"
            )

        if logging["requires_review"]:
            review_reasons.append(
                "logging_configuration"
            )

        review_required = bool(review_reasons)

        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "metadata_supplied": bool(self.metadata),
            "monitoring_evidence": evidence,
            "evidence_coverage": coverage,
            "missing_monitoring_fields": missing_fields,
            "additional_metadata_fields": additional_fields,
            "logging_analysis": logging,
            "monitoring_history_analysis": monitoring_history,
            "review_required": review_required,
            "review_reasons": review_reasons,
        }
