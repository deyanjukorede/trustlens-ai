
"""
End-to-end integration tests for TrustLens AI.

Validate the installed command-line assessment workflow,
JSON report generation, five-dimension reporting,
input validation, and protection against overwriting
existing reports.

These tests use synthetic data only.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATASET = PROJECT_ROOT / "examples" / "sample_dataset.csv"


def run_cli(*arguments):
    """Execute TrustLens CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "trustlens.cli", *map(str, arguments)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_end_to_end_basic_assessment(tmp_path):
    """A CSV assessment must generate a valid five-dimension report."""
    output = tmp_path / "basic_report.json"

    result = run_cli(
        "assess",
        SAMPLE_DATASET,
        "--output",
        output,
    )

    assert result.returncode == 0, (
        f"CLI failed:\n{result.stdout}\n{result.stderr}"
    )

    assert output.is_file()

    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["framework"] == "TrustLens AI"
    assert isinstance(report["dimensions"], dict)
    assert len(report["dimensions"]) == 5

    expected_dimensions = {
        "data_quality",
        "data_governance",
        "ai_readiness",
        "responsible_ai",
        "operational_trust",
    }

    assert expected_dimensions.issubset(
        set(report["dimensions"])
    )

    assert "cross_dimension_summary" in report


def test_end_to_end_contextual_assessment(tmp_path):
    """Optional modelling context must be accepted by the CLI."""
    output = tmp_path / "contextual_report.json"

    result = run_cli(
        "assess",
        SAMPLE_DATASET,
        "--output",
        output,
        "--target",
        "target",
        "--prediction",
        "prediction",
        "--sensitive",
        "group",
        "--identifier",
        "record_id",
    )

    assert result.returncode == 0, (
        f"Contextual assessment failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert len(report["dimensions"]) == 5
    assert "cross_dimension_summary" in report


def test_end_to_end_reference_dataset(tmp_path):
    """Reference data must be accepted for drift analysis."""
    output = tmp_path / "reference_report.json"

    result = run_cli(
        "assess",
        SAMPLE_DATASET,
        "--reference",
        SAMPLE_DATASET,
        "--output",
        output,
    )

    assert result.returncode == 0, (
        f"Reference assessment failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert "operational_trust" in report["dimensions"]


def test_end_to_end_governance_metadata(tmp_path):
    """The CLI must accept a valid governance metadata JSON file."""
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "metadata_report.json"

    metadata.write_text(
        json.dumps(
            {
                "dataset_name": "Synthetic TrustLens Dataset",
                "description": "End-to-end test dataset",
                "data_owner": "Test Steward",
                "source": "Synthetic data",
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "assess",
        SAMPLE_DATASET,
        "--metadata",
        metadata,
        "--output",
        output,
    )

    assert result.returncode == 0, (
        f"Metadata assessment failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert "data_governance" in report["dimensions"]


def test_end_to_end_missing_dataset(tmp_path):
    """Missing input files must produce a non-zero exit status."""
    missing_dataset = tmp_path / "missing.csv"
    output = tmp_path / "report.json"

    result = run_cli(
        "assess",
        missing_dataset,
        "--output",
        output,
    )

    assert result.returncode != 0
    assert not output.exists()
    assert "error" in result.stderr.lower()


def test_end_to_end_invalid_target_column(tmp_path):
    """Unknown prediction targets must be rejected."""
    output = tmp_path / "invalid_target_report.json"

    result = run_cli(
        "assess",
        SAMPLE_DATASET,
        "--target",
        "column_that_does_not_exist",
        "--output",
        output,
    )

    assert result.returncode != 0
    assert not output.exists()
    assert "target" in result.stderr.lower()


def test_end_to_end_invalid_metadata(tmp_path):
    """Invalid metadata JSON must be rejected."""
    metadata = tmp_path / "invalid_metadata.json"
    output = tmp_path / "invalid_metadata_report.json"

    metadata.write_text(
        "{this is not valid json}",
        encoding="utf-8",
    )

    result = run_cli(
        "assess",
        SAMPLE_DATASET,
        "--metadata",
        metadata,
        "--output",
        output,
    )

    assert result.returncode != 0
    assert not output.exists()
    assert "error" in result.stderr.lower()


def test_end_to_end_prevent_report_overwrite(tmp_path):
    """Existing JSON reports must never be overwritten."""
    output = tmp_path / "existing_report.json"
    original_content = '{"existing": true}\n'

    output.write_text(
        original_content,
        encoding="utf-8",
    )

    result = run_cli(
        "assess",
        SAMPLE_DATASET,
        "--output",
        output,
    )

    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == original_content
    assert "overwrite" in result.stderr.lower()


def test_end_to_end_cli_help():
    """The CLI must expose usable help information."""
    result = run_cli("--help")

    assert result.returncode == 0
    assert "assess" in result.stdout.lower()

    assess_result = run_cli("assess", "--help")

    assert assess_result.returncode == 0
    assert "--output" in assess_result.stdout
    assert "--target" in assess_result.stdout


def test_end_to_end_json_report_is_parseable(tmp_path):
    """Generated JSON must be valid and contain no nonstandard NaN."""
    output = tmp_path / "validated_report.json"

    result = run_cli(
        "assess",
        SAMPLE_DATASET,
        "--output",
        output,
    )

    assert result.returncode == 0, result.stderr

    raw_report = output.read_text(encoding="utf-8")

    def reject_nonstandard_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    report = json.loads(
        raw_report,
        parse_constant=reject_nonstandard_constant,
    )

    assert isinstance(report, dict)
    assert report["framework"] == "TrustLens AI"
