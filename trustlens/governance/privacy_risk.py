"""
Privacy-risk assessment for TrustLens AI.

This module evaluates the privacy risk associated with sensitive
information detected in a dataset. It builds on TrustLens sensitive-data
detection and converts detection results into an interpretable privacy
risk score and risk level.

The assessment is intended to support data-governance review. It does
not determine legal or regulatory compliance.
"""

from typing import Any, Dict, List

import pandas as pd

from .sensitive_data import SensitiveDataDetector


class PrivacyRiskAssessor:
    """
    Assess dataset privacy risk using detected sensitive information.

    Privacy risk is evaluated using:

    - the number of sensitive columns detected;
    - the proportion of dataset columns containing sensitive data;
    - the types of sensitive information detected; and
    - configurable risk weights for different sensitive-data types.

    The resulting score ranges from 0 to 100, where a higher score
    represents greater potential privacy exposure.
    """

    DEFAULT_RISK_WEIGHTS = {
        "email": 20,
        "phone": 20,
        "ip_address": 15,
        "name": 15,
        "address": 20,
        "date_of_birth": 25,
        "national_id": 30,
        "passport": 30,
        "financial": 30,
        "health": 30,
        "biometric": 35,
        "location": 20,
        "other_sensitive": 10,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        risk_weights: Dict[str, int] | None = None,
    ) -> None:
        """
        Initialise the privacy-risk assessor.

        Parameters
        ----------
        data:
            Dataset to assess.
        risk_weights:
            Optional custom risk weights for sensitive-data types.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame.
        ValueError
            If the DataFrame is empty or custom weights are invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        self.data = data
        self.risk_weights = dict(self.DEFAULT_RISK_WEIGHTS)

        if risk_weights is not None:
            self._validate_risk_weights(risk_weights)
            self.risk_weights.update(risk_weights)

    @staticmethod
    def _validate_risk_weights(risk_weights: Dict[str, int]) -> None:
        """Validate custom privacy-risk weights."""
        if not isinstance(risk_weights, dict):
            raise TypeError("risk_weights must be a dictionary")

        for sensitive_type, weight in risk_weights.items():
            if not isinstance(sensitive_type, str):
                raise TypeError("risk weight keys must be strings")

            if not isinstance(weight, (int, float)):
                raise TypeError("risk weight values must be numeric")

            if weight < 0 or weight > 100:
                raise ValueError(
                    "risk weight values must be between 0 and 100"
                )

    def sensitive_data_summary(self) -> Dict[str, Any]:
        """
        Return the TrustLens sensitive-data detection summary.

        Returns
        -------
        dict
            Structured sensitive-data assessment.
        """
        detector = SensitiveDataDetector(self.data)
        return detector.summary()

    def detected_sensitive_types(self) -> List[str]:
        """
        Return unique sensitive-data types detected in the dataset.

        Returns
        -------
        list
            Sorted sensitive-data type names.
        """
        summary = self.sensitive_data_summary()
        detected_types = summary.get("sensitive_types_detected", [])

        return sorted(set(detected_types))

    def sensitive_column_ratio(self) -> float:
        """
        Calculate the proportion of columns identified as sensitive.

        Returns
        -------
        float
            Ratio between 0.0 and 1.0.
        """
        summary = self.sensitive_data_summary()
        sensitive_columns = summary.get("sensitive_columns_detected", 0)
        total_columns = len(self.data.columns)

        if total_columns == 0:
            return 0.0

        return float(sensitive_columns / total_columns)

    def type_risk_score(self) -> float:
        """
        Calculate risk contributed by detected sensitive-data types.

        Unknown sensitive-data types use the ``other_sensitive`` weight.

        Returns
        -------
        float
            Type-based privacy-risk score capped at 70.
        """
        detected_types = self.detected_sensitive_types()

        if not detected_types:
            return 0.0

        default_weight = self.risk_weights["other_sensitive"]

        total_weight = sum(
            self.risk_weights.get(sensitive_type, default_weight)
            for sensitive_type in detected_types
        )

        return float(min(total_weight, 70))

    def exposure_score(self) -> float:
        """
        Calculate privacy exposure from sensitive-column concentration.

        The sensitive-column ratio contributes up to 30 points.

        Returns
        -------
        float
            Exposure score between 0 and 30.
        """
        return float(min(self.sensitive_column_ratio() * 30, 30))

    def privacy_risk_score(self) -> float:
        """
        Calculate the overall privacy-risk score.

        The score combines sensitive-type risk and dataset exposure.

        Returns
        -------
        float
            Privacy-risk score between 0 and 100.
        """
        score = self.type_risk_score() + self.exposure_score()
        return round(min(score, 100.0), 2)

    def risk_level(self) -> str:
        """
        Convert the privacy-risk score into an interpretable level.

        Returns
        -------
        str
            One of ``low``, ``moderate``, ``high``, or ``critical``.
        """
        score = self.privacy_risk_score()

        if score < 25:
            return "low"

        if score < 50:
            return "moderate"

        if score < 75:
            return "high"

        return "critical"

    def recommendations(self) -> List[str]:
        """
        Generate governance recommendations from the privacy-risk result.

        Returns
        -------
        list
            Privacy and governance recommendations.
        """
        score = self.privacy_risk_score()

        if score == 0:
            return [
                "No sensitive data was detected by the current rules. "
                "Continue routine governance and privacy monitoring."
            ]

        recommendations = [
            "Review detected sensitive fields and confirm that their "
            "collection is necessary for the intended purpose.",
            "Apply appropriate access controls to sensitive information.",
            "Consider masking, pseudonymisation, anonymisation, or removal "
            "of sensitive fields where appropriate.",
        ]

        if score >= 50:
            recommendations.append(
                "Perform a more detailed privacy review before using this "
                "dataset for AI or machine-learning workloads."
            )

        if score >= 75:
            recommendations.append(
                "Consider formal privacy-impact assessment and enhanced "
                "governance approval before production use."
            )

        return recommendations

    def assess(self) -> Dict[str, Any]:
        """
        Run the complete TrustLens privacy-risk assessment.

        Returns
        -------
        dict
            Structured privacy-risk assessment.
        """
        summary = self.sensitive_data_summary()

        return {
            "privacy_risk_score": self.privacy_risk_score(),
            "risk_level": self.risk_level(),
            "sensitive_columns_detected": summary.get(
                "sensitive_columns_detected",
                0,
            ),
            "sensitive_column_ratio": round(
                self.sensitive_column_ratio(),
                4,
            ),
            "sensitive_types_detected": self.detected_sensitive_types(),
            "type_risk_score": self.type_risk_score(),
            "exposure_score": round(self.exposure_score(), 2),
            "recommendations": self.recommendations(),
            "sensitive_data": summary,
        }
