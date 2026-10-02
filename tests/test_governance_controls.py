import pytest

from trustlens.governance.controls import GovernanceControlsAssessor


@pytest.fixture
def complete_controls():
    """Create a complete set of governance controls."""
    return {
        "data_owner": "Customer Operations",
        "approved_purpose": "Customer risk assessment",
        "data_classification": "Confidential",
        "access_control": True,
        "retention_policy": "7 years",
        "review_process": "Annual governance review",
        "accountability": "Head of Data",
    }


@pytest.fixture
def partial_controls():
    """Create a partially completed set of governance controls."""
    return {
        "data_owner": "Customer Operations",
        "approved_purpose": "Customer risk assessment",
        "data_classification": "Confidential",
        "access_control": True,
    }


def test_assessor_accepts_valid_controls(complete_controls):
    """Assessor should accept a valid governance controls mapping."""
    assessor = GovernanceControlsAssessor(complete_controls)

    assert assessor.controls == complete_controls


def test_assessor_accepts_no_controls():
    """Assessor should allow controls to be omitted."""
    assessor = GovernanceControlsAssessor()

    assert assessor.controls == {}


def test_invalid_controls_input():
    """Assessor should reject controls that are not a mapping."""
    with pytest.raises(TypeError):
        GovernanceControlsAssessor(["data_owner", "access_control"])


def test_required_controls_are_defined():
    """Assessor should expose the required governance controls."""
    assessor = GovernanceControlsAssessor()

    assert list(assessor.REQUIRED_CONTROLS) == [
        "data_owner",
        "approved_purpose",
        "data_classification",
        "access_control",
        "retention_policy",
        "review_process",
        "accountability",
    ]


def test_complete_controls_status(complete_controls):
    """Complete controls should all be reported as implemented."""
    assessor = GovernanceControlsAssessor(complete_controls)
    status = assessor.control_status()

    assert len(status) == 7
    assert all(
        details["implemented"]
        for details in status.values()
    )


def test_missing_controls_status():
    """Missing controls should all be reported as not implemented."""
    assessor = GovernanceControlsAssessor()
    status = assessor.control_status()

    assert len(status) == 7
    assert all(
        details["implemented"] is False
        for details in status.values()
    )


def test_control_status_preserves_values(complete_controls):
    """Control status should retain the supplied governance values."""
    assessor = GovernanceControlsAssessor(complete_controls)
    status = assessor.control_status()

    assert status["data_owner"]["value"] == "Customer Operations"
    assert status["access_control"]["value"] is True
    assert status["retention_policy"]["value"] == "7 years"


def test_complete_controls_score_100(complete_controls):
    """Complete governance controls should score 100."""
    assessor = GovernanceControlsAssessor(complete_controls)

    assert assessor.coverage_score() == 100.0
    assert assessor.governance_level() == "Complete"


def test_partial_controls_score(partial_controls):
    """Four of seven implemented controls should score 57.14."""
    assessor = GovernanceControlsAssessor(partial_controls)

    assert assessor.coverage_score() == 57.14
    assert assessor.governance_level() == "Partial"


def test_missing_controls_score_zero():
    """No implemented governance controls should score zero."""
    assessor = GovernanceControlsAssessor()

    assert assessor.coverage_score() == 0.0
    assert assessor.governance_level() == "Missing"


def test_strong_governance_level():
    """Six of seven implemented controls should be classified Strong."""
    controls = {
        "data_owner": "Customer Operations",
        "approved_purpose": "Customer risk assessment",
        "data_classification": "Confidential",
        "access_control": True,
        "retention_policy": "7 years",
        "review_process": "Annual governance review",
    }

    assessor = GovernanceControlsAssessor(controls)

    assert assessor.coverage_score() == 85.71
    assert assessor.governance_level() == "Strong"


def test_limited_governance_level():
    """One implemented control should be classified Limited."""
    assessor = GovernanceControlsAssessor(
        {
            "data_owner": "Customer Operations",
        }
    )

    assert assessor.coverage_score() == 14.29
    assert assessor.governance_level() == "Limited"


def test_implemented_controls(complete_controls):
    """Implemented controls should be returned correctly."""
    assessor = GovernanceControlsAssessor(complete_controls)

    assert assessor.implemented_controls() == [
        "data_owner",
        "approved_purpose",
        "data_classification",
        "access_control",
        "retention_policy",
        "review_process",
        "accountability",
    ]


def test_missing_controls(partial_controls):
    """Missing controls should be identified correctly."""
    assessor = GovernanceControlsAssessor(partial_controls)

    assert assessor.missing_controls() == [
        "retention_policy",
        "review_process",
        "accountability",
    ]


def test_none_control_is_missing(complete_controls):
    """None should be treated as a missing governance control."""
    controls = dict(complete_controls)
    controls["data_owner"] = None

    assessor = GovernanceControlsAssessor(controls)

    assert "data_owner" in assessor.missing_controls()


def test_false_control_is_missing(complete_controls):
    """False should be treated as a missing governance control."""
    controls = dict(complete_controls)
    controls["access_control"] = False

    assessor = GovernanceControlsAssessor(controls)

    assert "access_control" in assessor.missing_controls()


def test_empty_string_control_is_missing(complete_controls):
    """An empty string should be treated as a missing control."""
    controls = dict(complete_controls)
    controls["data_owner"] = ""

    assessor = GovernanceControlsAssessor(controls)

    assert "data_owner" in assessor.missing_controls()


def test_whitespace_control_is_missing(complete_controls):
    """Whitespace-only text should be treated as a missing control."""
    controls = dict(complete_controls)
    controls["retention_policy"] = "   "

    assessor = GovernanceControlsAssessor(controls)

    assert "retention_policy" in assessor.missing_controls()


def test_recommendations_for_missing_controls(partial_controls):
    """Missing controls should generate governance recommendations."""
    assessor = GovernanceControlsAssessor(partial_controls)
    recommendations = assessor.recommendations()

    assert len(recommendations) == 3
    assert (
        "Define and document an appropriate data retention policy."
        in recommendations
    )
    assert (
        "Establish a documented process for periodic governance review."
        in recommendations
    )
    assert (
        "Document who is accountable for governance decisions "
        "relating to the dataset."
        in recommendations
    )


def test_complete_controls_recommendation(complete_controls):
    """Complete controls should return a positive recommendation."""
    assessor = GovernanceControlsAssessor(complete_controls)

    assert assessor.recommendations() == [
        "All required governance controls are documented."
    ]


def test_assess_returns_expected_structure(complete_controls):
    """Assessment should contain all expected report sections."""
    assessor = GovernanceControlsAssessor(complete_controls)
    report = assessor.assess()

    expected_keys = {
        "governance_controls_score",
        "governance_controls_level",
        "controls_assessed",
        "controls_implemented",
        "controls_missing",
        "required_controls",
        "implemented_controls",
        "missing_controls",
        "control_status",
        "recommendations",
    }

    assert set(report.keys()) == expected_keys


def test_assess_reports_complete_counts(complete_controls):
    """Assessment should report complete control counts correctly."""
    assessor = GovernanceControlsAssessor(complete_controls)
    report = assessor.assess()

    assert report["controls_assessed"] == 7
    assert report["controls_implemented"] == 7
    assert report["controls_missing"] == 0


def test_assess_reports_partial_counts(partial_controls):
    """Assessment should report partial control counts correctly."""
    assessor = GovernanceControlsAssessor(partial_controls)
    report = assessor.assess()

    assert report["controls_assessed"] == 7
    assert report["controls_implemented"] == 4
    assert report["controls_missing"] == 3


def test_assessment_is_deterministic(partial_controls):
    """Repeated assessments should produce identical results."""
    assessor = GovernanceControlsAssessor(partial_controls)

    first_report = assessor.assess()
    second_report = assessor.assess()

    assert first_report == second_report
