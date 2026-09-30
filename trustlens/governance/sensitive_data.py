"""
Sensitive data detection for TrustLens AI.

This module provides rule-based detection of potentially sensitive
and personally identifiable information (PII) in pandas DataFrames.

Detection combines column-name indicators with value-pattern analysis
to provide a transparent and extensible foundation for privacy and
data-governance assessment.
"""

import re
from typing import Any, Dict, List, Set

import pandas as pd


class SensitiveDataDetector:
    """
    Detect potentially sensitive data within a pandas DataFrame.

    The detector evaluates both column names and sampled column values.
    It is intentionally rule-based so that detection decisions remain
    transparent, explainable, and easy to extend.
    """

    COLUMN_INDICATORS: Dict[str, Set[str]] = {
        "email": {
            "email",
            "email_address",
            "emailaddress",
            "e_mail",
        },
        "phone_number": {
            "phone",
            "phone_number",
            "phonenumber",
            "mobile",
            "mobile_number",
            "telephone",
            "tel",
        },
        "name": {
            "name",
            "full_name",
            "fullname",
            "first_name",
            "firstname",
            "last_name",
            "lastname",
            "surname",
        },
        "address": {
            "address",
            "home_address",
            "residential_address",
            "street_address",
        },
        "date_of_birth": {
            "date_of_birth",
            "dateofbirth",
            "dob",
            "birth_date",
            "birthdate",
        },
        "national_identifier": {
            "national_id",
            "national_identifier",
            "nin",
            "ssn",
            "social_security_number",
            "passport_number",
            "passport_no",
        },
        "financial_information": {
            "bank_account",
            "bank_account_number",
            "account_number",
            "credit_card",
            "credit_card_number",
            "card_number",
            "iban",
        },
        "ip_address": {
            "ip",
            "ip_address",
            "ipaddress",
        },
    }

    VALUE_PATTERNS: Dict[str, re.Pattern] = {
        "email": re.compile(
            r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        ),
        "ip_address": re.compile(
            r"^(?:\d{1,3}\.){3}\d{1,3}$"
        ),
        "phone_number": re.compile(
            r"^\+?[\d\s().-]{7,20}$"
        ),
    }

    def __init__(self, data: pd.DataFrame, sample_size: int = 100) -> None:
        """
        Initialise the sensitive-data detector.

        Parameters
        ----------
        data:
            Dataset to inspect.
        sample_size:
            Maximum number of non-missing values inspected per column.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame.
        ValueError
            If the DataFrame is empty or sample_size is invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("data must not be empty")

        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")

        self.data = data
        self.sample_size = sample_size

    @staticmethod
    def _normalise_column_name(column: Any) -> str:
        """Normalise a column name for comparison with indicators."""
        name = str(column).strip().lower()
        name = re.sub(r"[\s\-]+", "_", name)
        name = re.sub(r"_+", "_", name)
        return name

    def detect_by_column_name(self) -> Dict[str, List[str]]:
        """
        Detect sensitive fields using column-name indicators.

        Returns
        -------
        dict
            Mapping of column names to detected sensitive-data types.
        """
        detected: Dict[str, List[str]] = {}

        for column in self.data.columns:
            normalised = self._normalise_column_name(column)
            matches: List[str] = []

            for sensitive_type, indicators in self.COLUMN_INDICATORS.items():
                if normalised in indicators:
                    matches.append(sensitive_type)

            if matches:
                detected[str(column)] = sorted(matches)

        return detected

    def _value_match_rate(
        self,
        series: pd.Series,
        pattern: re.Pattern,
    ) -> float:
        """
        Calculate the proportion of sampled values matching a pattern.
        """
        values = series.dropna().head(self.sample_size)

        if values.empty:
            return 0.0

        matches = 0

        for value in values:
            if pattern.fullmatch(str(value).strip()):
                matches += 1

        return matches / len(values)

    def detect_by_value_pattern(
        self,
        threshold: float = 0.8,
    ) -> Dict[str, List[str]]:
        """
        Detect sensitive fields using value-pattern analysis.

        A sensitive type is reported when at least ``threshold`` of
        sampled non-missing values match its configured pattern.

        Parameters
        ----------
        threshold:
            Minimum match rate required for detection.

        Returns
        -------
        dict
            Mapping of column names to detected sensitive-data types.
        """
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric")

        if not 0 < threshold <= 1:
            raise ValueError("threshold must be greater than 0 and at most 1")

        detected: Dict[str, List[str]] = {}

        for column in self.data.columns:
            matches: List[str] = []

            for sensitive_type, pattern in self.VALUE_PATTERNS.items():
                match_rate = self._value_match_rate(
                    self.data[column],
                    pattern,
                )

                if match_rate >= threshold:
                    matches.append(sensitive_type)

            if matches:
                detected[str(column)] = sorted(matches)

        return detected

    def detect(self) -> Dict[str, Dict[str, Any]]:
        """
        Combine column-name and value-pattern detection.

        Returns
        -------
        dict
            Mapping of detected columns to their sensitive-data
            classifications and detection sources.
        """
        name_matches = self.detect_by_column_name()
        value_matches = self.detect_by_value_pattern()

        detected_columns = set(name_matches) | set(value_matches)
        results: Dict[str, Dict[str, Any]] = {}

        for column in sorted(detected_columns):
            sensitive_types = set(name_matches.get(column, []))
            sensitive_types.update(value_matches.get(column, []))

            detection_sources: List[str] = []

            if column in name_matches:
                detection_sources.append("column_name")

            if column in value_matches:
                detection_sources.append("value_pattern")

            results[column] = {
                "sensitive_types": sorted(sensitive_types),
                "detection_sources": detection_sources,
            }

        return results

    def summary(self) -> Dict[str, Any]:
        """
        Return a concise sensitive-data assessment summary.

        Returns
        -------
        dict
            Summary of detected sensitive-data fields.
        """
        detected = self.detect()

        sensitive_types = sorted(
            {
                sensitive_type
                for details in detected.values()
                for sensitive_type in details["sensitive_types"]
            }
        )

        return {
            "sensitive_columns_detected": len(detected),
            "sensitive_columns": sorted(detected.keys()),
            "sensitive_types_detected": sensitive_types,
            "details": detected,
        }
