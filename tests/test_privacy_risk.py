import pandas as pd
import pytest

from trustlens.governance.privacy_risk import PrivacyRiskAssessor


@pytest.fixture
def sensitive_data():
    """Create a representative dataset containing sensitive information."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "full_name": [
                "Ada Smith",
                "John Brown",
                "Mary Jones",
                "Peter White",
            ],
            "email": [
                "ada@example.com",
                "john@example.com",
                "mary@example.com",
                "peter@example.com",
            ],
            "mobile_number": [
                "+447700900001",
                "+447700900002",
                "+447700900003",
                "+447700900004",
            ],
            "age": [28, 35, 42, 31],
        }
    )


@pytest.fixture
def ordinary_data():
    """Create a dataset without obvious sensitive information."""
    return pd.DataFrame(
        {
            "product": [
                "Laptop",
                "Monitor",
                "Keyboard",
                "Mouse",
            ],
            "quantity": [2, 5, 10, 4],
            "category": [
                "hardware",
                "hardware",
                "accessory",
                "accessory",
            ],
        }
    )


def test_privacy_risk_assessor_accepts_dataframe(sensitive_data):
    """Assessor should accept a valid pandas DataFrame."""
    assessor = PrivacyRiskAssessor(sensitive_data)

    assert assessor.data is sensitive_data


def test_invalid_input():
    """Assessor should reject input that is not a pandas DataFrame."""
    with pytest.raises(TypeError):
        PrivacyRiskAssessor([1, 2, 3])


def test_empty_dataframe():
    """Assessor should reject an empty pandas DataFrame."""
    with pytest.raises(ValueError):
        PrivacyRiskAssessor(pd.DataFrame())


def test_sensitive_data_summary(sensitive_data):
    """Sensitive-data summary should be available to privacy assessment."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    summary = assessor.sensitive_data_summary()

    assert isinstance(summary, dict)
    assert summary["sensitive_columns_detected"] >= 1


def test_sensitive_column_ratio(sensitive_data):
    """Sensitive-column ratio should be between zero and one."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    ratio = assessor.sensitive_column_ratio()

    assert 0.0 <= ratio <= 1.0
    assert ratio > 0.0


def test_ordinary_data_has_lower_sensitive_ratio(
    sensitive_data,
    ordinary_data,
):
    """Ordinary data should have a lower sensitive-column ratio."""
    sensitive_assessor = PrivacyRiskAssessor(sensitive_data)
    ordinary_assessor = PrivacyRiskAssessor(ordinary_data)

    assert (
        ordinary_assessor.sensitive_column_ratio()
        < sensitive_assessor.sensitive_column_ratio()
    )


def test_detected_sensitive_types(sensitive_data):
    """Assessor should identify detected sensitive-data types."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    detected_types = assessor.detected_sensitive_types()

    assert isinstance(detected_types, list)
    assert len(detected_types) >= 1


def test_type_risk_score_is_numeric(sensitive_data):
    """Sensitive-data type risk score should be numeric and bounded."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    score = assessor.type_risk_score()

    assert isinstance(score, (int, float))
    assert 0.0 <= score <= 100.0


def test_exposure_score_is_numeric(sensitive_data):
    """Exposure score should be numeric and bounded."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    score = assessor.exposure_score()

    assert isinstance(score, (int, float))
    assert 0.0 <= score <= 100.0


def test_privacy_risk_score_is_numeric(sensitive_data):
    """Overall privacy-risk score should be numeric and bounded."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    score = assessor.privacy_risk_score()

    assert isinstance(score, (int, float))
    assert 0.0 <= score <= 100.0


def test_sensitive_data_has_privacy_risk(sensitive_data):
    """Dataset containing sensitive information should have privacy risk."""
    assessor = PrivacyRiskAssessor(sensitive_data)

    assert assessor.privacy_risk_score() > 0.0


def test_ordinary_data_has_lower_privacy_risk(
    sensitive_data,
    ordinary_data,
):
    """Ordinary data should have lower risk than sensitive data."""
    sensitive_assessor = PrivacyRiskAssessor(sensitive_data)
    ordinary_assessor = PrivacyRiskAssessor(ordinary_data)

    assert (
        ordinary_assessor.privacy_risk_score()
        < sensitive_assessor.privacy_risk_score()
    )


def test_risk_level_is_supported_value(sensitive_data):
    """Risk level should use one of the supported classifications."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    risk_level = assessor.risk_level()

    assert risk_level in {
        "low",
        "moderate",
        "high",
        "critical",
    }


def test_recommendations_return_list(sensitive_data):
    """Privacy assessment should generate governance recommendations."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    recommendations = assessor.recommendations()

    assert isinstance(recommendations, list)
    assert len(recommendations) >= 1


def test_assess_returns_expected_sections(sensitive_data):
    """Complete assessment should contain all privacy-risk sections."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    expected_sections = {
        "privacy_risk_score",
        "risk_level",
        "sensitive_columns_detected",
        "sensitive_column_ratio",
        "sensitive_types_detected",
        "type_risk_score",
        "exposure_score",
        "recommendations",
        "sensitive_data",
    }

    assert expected_sections.issubset(report.keys())


def test_assess_privacy_score_matches_method(sensitive_data):
    """Assessment score should match the standalone scoring method."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    assert report["privacy_risk_score"] == assessor.privacy_risk_score()


def test_assess_risk_level_matches_method(sensitive_data):
    """Assessment risk level should match the standalone method."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    assert report["risk_level"] == assessor.risk_level()


def test_assess_sensitive_ratio_matches_method(sensitive_data):
    """Assessment should contain the calculated sensitive-column ratio."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    assert report["sensitive_column_ratio"] == round(
        assessor.sensitive_column_ratio(),
        4,
    )


def test_assess_type_score_matches_method(sensitive_data):
    """Assessment should contain the calculated type-risk score."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    assert report["type_risk_score"] == assessor.type_risk_score()


def test_assess_exposure_score_matches_method(sensitive_data):
    """Assessment should contain the calculated exposure score."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    assert report["exposure_score"] == round(
        assessor.exposure_score(),
        2,
    )


def test_assess_recommendations_match_method(sensitive_data):
    """Assessment should contain generated recommendations."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    assert report["recommendations"] == assessor.recommendations()


def test_assess_contains_sensitive_data_summary(sensitive_data):
    """Assessment should include underlying sensitive-data analysis."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    assert report["sensitive_data"] == assessor.sensitive_data_summary()


def test_privacy_score_is_deterministic(sensitive_data):
    """Repeated assessment of the same data should return the same score."""
    assessor = PrivacyRiskAssessor(sensitive_data)

    first_score = assessor.privacy_risk_score()
    second_score = assessor.privacy_risk_score()

    assert first_score == second_score


def test_complete_assessment_is_dictionary(sensitive_data):
    """Complete privacy-risk assessment should return a dictionary."""
    assessor = PrivacyRiskAssessor(sensitive_data)
    report = assessor.assess()

    assert isinstance(report, dict)
