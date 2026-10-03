"""
Reproducibility and reliability readiness analysis for TrustLens AI.

This module evaluates evidence that can support reproducible and
operationally reliable data and AI workflows.

Reproducibility cannot be established from a dataset alone. The
analyzer therefore considers both dataset characteristics and optional
workflow metadata such as dataset versions, code versions, random
seeds, environment information, lineage, and execution identifiers.

Missing evidence is reported as a readiness or review indicator. It
does not independently establish that a workflow is unreliable,
irreproducible, unsafe, or non-compliant.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


class ReproducibilityReadinessAnalyzer:
    """
    Assess evidence supporting reproducibility and reliability readiness.

    Parameters
    ----------
    data:
        Current dataset being assessed.
    metadata:
        Optional dictionary describing reproducibility evidence.

        Recognised fields are:

        - ``dataset_version``
        - ``code_version``
        - ``random_seed``
        - ``environment``
        - ``data_source``
        - ``lineage``
        - ``execution_id``

        Additional metadata fields are preserved but do not currently
        affect readiness indicators.
    identifier_column:
        Optional dataset column intended to provide row-level
        identification.
    """

    REQUIRED_METADATA_FIELDS = (
        "dataset_version",
        "code_version",
        "random_seed",
        "environment",
        "data_source",
        "lineage",
        "execution_id",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        identifier_column: Optional[str] = None,
    ) -> None:
        """Initialise the reproducibility readiness analyzer."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary when supplied")

        if identifier_column is not None:
            if not isinstance(identifier_column, str):
                raise TypeError("identifier_column must be a string")

            if not identifier_column.strip():
                raise ValueError(
                    "identifier_column must not be empty"
                )

            if identifier_column not in data.columns:
                raise ValueError(
                    "identifier_column must exist in data"
                )

        self.data = data.copy()
        self.metadata = dict(metadata) if metadata is not None else {}
        self.identifier_column = identifier_column

        self.row_count = int(len(self.data))
        self.column_count = int(len(self.data.columns))

    @staticmethod
    def _has_evidence(value: Any) -> bool:
        """
        Return whether a metadata value contains usable evidence.

        Numeric zero and boolean values are treated as valid evidence,
        which is important for values such as ``random_seed=0``.
        """
        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)

        return True

    def metadata_evidence(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate recognised reproducibility metadata fields."""
        evidence: Dict[str, Dict[str, Any]] = {}

        for field in self.REQUIRED_METADATA_FIELDS:
            value = self.metadata.get(field)

            evidence[field] = {
                "provided": self._has_evidence(value),
                "value": value,
            }

        return evidence

    def missing_metadata_fields(self) -> List[str]:
        """Return recognised evidence fields that are not supplied."""
        evidence = self.metadata_evidence()

        return [
            field
            for field in self.REQUIRED_METADATA_FIELDS
            if not evidence[field]["provided"]
        ]

    def additional_metadata_fields(self) -> List[str]:
        """Return supplied metadata fields outside the recognised set."""
        return [
            field
            for field in self.metadata
            if field not in self.REQUIRED_METADATA_FIELDS
        ]

    def identifier_analysis(self) -> Dict[str, Any]:
        """Evaluate optional row-identifier evidence."""
        if self.identifier_column is None:
            return {
                "provided": False,
                "column": None,
                "missing_values": None,
                "duplicate_values": None,
                "unique_values": None,
                "is_complete": None,
                "is_unique": None,
                "requires_review": False,
            }

        series = self.data[self.identifier_column]

        missing_values = int(series.isna().sum())
        duplicate_values = int(
            series.dropna().duplicated().sum()
        )
        unique_values = int(series.dropna().nunique())

        is_complete = missing_values == 0
        is_unique = duplicate_values == 0

        return {
            "provided": True,
            "column": self.identifier_column,
            "missing_values": missing_values,
            "duplicate_values": duplicate_values,
            "unique_values": unique_values,
            "is_complete": is_complete,
            "is_unique": is_unique,
            "requires_review": (
                not is_complete or not is_unique
            ),
        }

    def metadata_coverage(self) -> Dict[str, Any]:
        """Summarise coverage of recognised reproducibility evidence."""
        evidence = self.metadata_evidence()

        provided_count = sum(
            1
            for result in evidence.values()
            if result["provided"]
        )

        total_fields = len(self.REQUIRED_METADATA_FIELDS)

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

    def assess(self) -> Dict[str, Any]:
        """Return the complete reproducibility readiness assessment."""
        evidence = self.metadata_evidence()
        missing_fields = self.missing_metadata_fields()
        additional_fields = self.additional_metadata_fields()
        identifier = self.identifier_analysis()
        coverage = self.metadata_coverage()

        review_reasons: List[str] = []

        if missing_fields:
            review_reasons.append(
                "missing_reproducibility_metadata"
            )

        if identifier["requires_review"]:
            review_reasons.append(
                "identifier_integrity"
            )

        review_required = bool(review_reasons)

        return {
            "dataset": {
                "rows": self.row_count,
                "columns": self.column_count,
            },
            "metadata_supplied": bool(self.metadata),
            "metadata_evidence": evidence,
            "metadata_coverage": coverage,
            "missing_metadata_fields": missing_fields,
            "additional_metadata_fields": additional_fields,
            "identifier_analysis": identifier,
            "review_required": review_required,
            "review_reasons": review_reasons,
        }
