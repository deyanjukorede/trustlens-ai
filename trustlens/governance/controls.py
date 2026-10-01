"""
Governance controls assessment for TrustLens AI.

This module evaluates whether important data governance controls are
documented for a dataset used in artificial intelligence and machine
learning systems.

The assessment focuses on practical governance controls including
ownership, approved purpose, data classification, access control,
retention, review arrangements, and accountability. It provides a
structured foundation for broader TrustLens governance scoring.
"""

from typing import Any, Dict, Mapping, Optional


class GovernanceControlsAssessor:
    """
    Assess documented governance controls associated with a dataset.

    The assessor evaluates a defined set of governance controls and
    reports their implementation status, overall coverage score,
    governance level, missing controls, and recommended actions.
    """

    REQUIRED_CONTROLS = (
        "data_owner",
        "approved_purpose",
        "data_classification",
        "access_control",
        "retention_policy",
        "review_process",
        "accountability",
    )

    def __init__(
        self,
        controls: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialise the governance controls assessor.

        Parameters
        ----------
        controls:
            Optional mapping containing governance control information.

            Example::

                {
                    "data_owner": "Customer Operations",
                    "approved_purpose": "Customer risk assessment",
                    "data_classification": "Confidential",
                    "access_control": True,
                    "retention_policy": "7 years",
                    "review_process": "Annual review",
                    "accountability": "Head of Data",
                }

        Raises
        ------
        TypeError
            If controls is not a mapping or None.
        """
        if controls is not None and not isinstance(controls, Mapping):
            raise TypeError("controls must be a mapping or None")

        self.controls = dict(controls or {})

    @staticmethod
    def _is_control_present(value: Any) -> bool:
        """
        Return whether a governance control should be treated as present.

        None, False, empty strings, and whitespace-only strings are
        considered missing. Other supplied values are considered present.
        """
        if value is None:
            return False

        if value is False:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        return True

    def control_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Assess the implementation status of each required control.

        Returns
        -------
        dict
            Mapping of governance control names to their supplied value
            and implementation status.
        """
        result: Dict[str, Dict[str, Any]] = {}

        for control in self.REQUIRED_CONTROLS:
            value = self.controls.get(control)

            result[control] = {
                "value": value,
                "implemented": self._is_control_present(value),
            }

        return result

    def implemented_controls(self) -> list[str]:
        """
        Return the names of implemented governance controls.
        """
        status = self.control_status()

        return [
            control
            for control, details in status.items()
            if details["implemented"]
        ]

    def missing_controls(self) -> list[str]:
        """
        Return the names of governance controls that are not implemented.
        """
        status = self.control_status()

        return [
            control
            for control, details in status.items()
            if not details["implemented"]
        ]

    def coverage_score(self) -> float:
        """
        Calculate the overall governance control coverage score.

        Returns
        -------
        float
            Percentage score between 0 and 100.
        """
        total_controls = len(self.REQUIRED_CONTROLS)

        if total_controls == 0:
            return 0.0

        implemented = len(self.implemented_controls())

        return round((implemented / total_controls) * 100, 2)

    def governance_level(self) -> str:
        """
        Classify the overall governance control coverage level.

        Returns
        -------
        str
            One of Complete, Strong, Partial, Limited, or Missing.
        """
        score = self.coverage_score()

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
        Generate recommendations for missing governance controls.

        Returns
        -------
        list of str
            Recommended governance actions.
        """
        missing = self.missing_controls()

        if not missing:
            return [
                "All required governance controls are documented."
            ]

        recommendation_map = {
            "data_owner": (
                "Assign and document an accountable data owner."
            ),
            "approved_purpose": (
                "Document the approved purpose for using the dataset."
            ),
            "data_classification": (
                "Classify the dataset according to the organisation's "
                "data classification policy."
            ),
            "access_control": (
                "Document and implement appropriate access controls."
            ),
            "retention_policy": (
                "Define and document an appropriate data retention policy."
            ),
            "review_process": (
                "Establish a documented process for periodic governance review."
            ),
            "accountability": (
                "Document who is accountable for governance decisions "
                "relating to the dataset."
            ),
        }

        return [
            recommendation_map[control]
            for control in missing
        ]

    def assess(self) -> Dict[str, Any]:
        """
        Run the complete governance controls assessment.

        Returns
        -------
        dict
            Structured governance controls assessment report.
        """
        status = self.control_status()
        implemented = self.implemented_controls()
        missing = self.missing_controls()

        return {
            "governance_controls_score": self.coverage_score(),
            "governance_controls_level": self.governance_level(),
            "controls_assessed": len(self.REQUIRED_CONTROLS),
            "controls_implemented": len(implemented),
            "controls_missing": len(missing),
            "required_controls": list(self.REQUIRED_CONTROLS),
            "implemented_controls": implemented,
            "missing_controls": missing,
            "control_status": status,
            "recommendations": self.recommendations(),
        }
