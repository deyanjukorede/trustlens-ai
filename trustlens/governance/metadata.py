"""
Metadata completeness assessment for TrustLens AI.

This module evaluates whether a dataset contains sufficient structural
metadata to support responsible governance and trustworthy AI use.

The assessment focuses on column-level metadata such as column names,
data types, descriptions, ownership, and business definitions. It is
designed to provide a reusable foundation for broader TrustLens data
governance scoring.
"""

from typing import Any, Dict, Mapping, Optional

import pandas as pd


class MetadataCompletenessAssessor:
    """
    Assess the completeness of metadata associated with a dataset.

    The assessor evaluates automatically available structural metadata
    and optional user-supplied metadata. Results are returned in a
    structured format that can later be integrated into the broader
    TrustLens governance assessment.
    """

    REQUIRED_METADATA_FIELDS = (
        "description",
        "owner",
        "business_definition",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        """
        Initialise the metadata completeness assessor.

        Parameters
        ----------
        data:
            Dataset whose metadata will be assessed.
        metadata:
            Optional mapping containing metadata for dataset columns.

            Example::

                {
                    "customer_id": {
                        "description": "Unique customer identifier",
                        "owner": "Customer Operations",
                        "business_definition": "Internal customer ID",
                    }
                }

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame or metadata is invalid.
        ValueError
            If the DataFrame is empty.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")

        self.data = data
        self.metadata = metadata or {}

    @staticmethod
    def _has_value(value: Any) -> bool:
        """
        Return whether a metadata value should be considered populated.

        None, empty strings, and whitespace-only strings are treated as
        missing metadata.
        """
        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        return True

    def structural_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata that can be derived directly from the DataFrame.

        Returns
        -------
        dict
            Mapping of column names to structural metadata.
        """
        result: Dict[str, Dict[str, Any]] = {}

        for column in self.data.columns:
            column_name = str(column)

            result[column_name] = {
                "column_name": column_name,
                "data_type": str(self.data[column].dtype),
            }

        return result

    def column_metadata_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Assess metadata completeness for every dataset column.

        Returns
        -------
        dict
            Column-level metadata completeness information.
        """
        result: Dict[str, Dict[str, Any]] = {}

        for column in self.data.columns:
            column_name = str(column)
            supplied_metadata = self.metadata.get(column_name, {})

            if not isinstance(supplied_metadata, Mapping):
                raise TypeError(
                    f"metadata for column '{column_name}' must be a mapping"
                )

            field_status = {
                field: self._has_value(supplied_metadata.get(field))
                for field in self.REQUIRED_METADATA_FIELDS
            }

            completed_fields = sum(field_status.values())
            total_fields = len(self.REQUIRED_METADATA_FIELDS)

            completeness_score = round(
                (completed_fields / total_fields) * 100,
                2,
            )

            missing_fields = [
                field
                for field, is_present in field_status.items()
                if not is_present
            ]

            result[column_name] = {
                "column_name": column_name,
                "data_type": str(self.data[column].dtype),
                "required_fields": list(self.REQUIRED_METADATA_FIELDS),
                "field_status": field_status,
                "completed_fields": completed_fields,
                "missing_fields": missing_fields,
                "completeness_score": completeness_score,
            }

        return result

    def completeness_score(self) -> float:
        """
        Calculate the overall metadata completeness score.

        Returns
        -------
        float
            Percentage score between 0 and 100.
        """
        status = self.column_metadata_status()

        if not status:
            return 0.0

        scores = [
            column_status["completeness_score"]
            for column_status in status.values()
        ]

        return round(sum(scores) / len(scores), 2)

    def completeness_level(self) -> str:
        """
        Classify the overall metadata completeness level.

        Returns
        -------
        str
            One of Complete, Strong, Partial, Limited, or Missing.
        """
        score = self.completeness_score()

        if score == 100:
            return "Complete"

        if score >= 75:
            return "Strong"

        if score >= 50:
            return "Partial"

        if score > 0:
            return "Limited"

        return "Missing"

    def recommendations(self) -> list[str]:
        """
        Generate governance recommendations for incomplete metadata.

        Returns
        -------
        list of str
            Recommended actions for improving metadata completeness.
        """
        status = self.column_metadata_status()
        recommendations = []

        for column_name, column_status in status.items():
            missing_fields = column_status["missing_fields"]

            if missing_fields:
                fields = ", ".join(missing_fields)
                recommendations.append(
                    f"Complete metadata for '{column_name}': {fields}."
                )

        if not recommendations:
            recommendations.append(
                "Metadata is complete for all assessed dataset columns."
            )

        return recommendations

    def assess(self) -> Dict[str, Any]:
        """
        Run the complete metadata completeness assessment.

        Returns
        -------
        dict
            Structured metadata assessment report.
        """
        status = self.column_metadata_status()

        columns_with_complete_metadata = sum(
            1
            for column_status in status.values()
            if column_status["completeness_score"] == 100
        )

        return {
            "metadata_completeness_score": self.completeness_score(),
            "metadata_completeness_level": self.completeness_level(),
            "columns_assessed": len(status),
            "columns_with_complete_metadata": (
                columns_with_complete_metadata
            ),
            "columns_with_incomplete_metadata": (
                len(status) - columns_with_complete_metadata
            ),
            "required_metadata_fields": list(
                self.REQUIRED_METADATA_FIELDS
            ),
            "structural_metadata": self.structural_metadata(),
            "column_metadata": status,
            "recommendations": self.recommendations(),
        }
