
"""
Integration tests for TrustLens AI documentation examples.

These tests verify that the published synthetic dataset and
unified Python example work together successfully.

They also check that the example reports all five assessment
dimensions and includes appropriate interpretation guidance.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATASET_PATH = EXAMPLES_DIR / "sample_dataset.csv"
EXAMPLE_PATH = EXAMPLES_DIR / "unified_assessment_example.py"


def test_documentation_sample_dataset_exists():
    """The documented synthetic CSV dataset must exist."""
    assert DATASET_PATH.is_file()


def test_documentation_sample_dataset_structure():
    """The sample dataset must contain the expected columns."""
    data = pd.read_csv(DATASET_PATH)

    expected_columns = {
        "record_id",
        "age",
        "income",
        "employment_status",
        "region",
        "group",
        "target",
        "prediction",
    }

    assert not data.empty
    assert len(data) == 20
    assert expected_columns.issubset(set(data.columns))
    assert data["record_id"].is_unique
    assert set(data["target"].dropna().unique()).issubset({0, 1})
    assert set(data["prediction"].dropna().unique()).issubset({0, 1})


def test_documentation_python_example_exists():
    """The documented unified assessment example must exist."""
    assert EXAMPLE_PATH.is_file()


def test_documentation_python_example_runs():
    """The published Python example must execute successfully."""
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_PATH),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        "The documented TrustLens example failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    output = result.stdout

    assert "TrustLens AI - Unified Assessment Example" in output
    assert "sample_dataset.csv" in output
    assert "Rows: 20" in output
    assert "Assessment completed." in output
    assert "Assessment dimensions:" in output
    assert "Cross-dimension summary:" in output

    # Confirm that the five assessment dimensions are represented.
    expected_dimensions = (
        "data_quality",
        "data_governance",
        "ai_readiness",
        "responsible_ai",
        "operational_trust",
    )

    for dimension in expected_dimensions:
        assert dimension in output, (
            f"Missing documented assessment dimension: {dimension}"
        )

    # The example must communicate that automated findings
    # require contextual interpretation and human review.
    assert "human review" in output.lower()
    assert "not a certification" in output.lower()


def test_quickstart_tutorial_exists_and_is_linked():
    """QUICKSTART.md must exist and be linked from the main README."""
    assert (PROJECT_ROOT / "QUICKSTART.md").is_file()
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "QUICKSTART.md" in readme


def test_quickstart_commands_produce_json_report(tmp_path):
    """Every CLI command in the quickstart must run and yield the documented report."""
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trustlens.cli",
            "assess",
            str(DATASET_PATH),
            "--output",
            str(report_path),
            "--target",
            "target",
            "--prediction",
            "prediction",
            "--sensitive",
            "group",
            "--identifier",
            "record_id",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        "The quickstart assessment command failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert report_path.is_file()

    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for dimension in (
        "data_quality",
        "data_governance",
        "ai_readiness",
        "responsible_ai",
        "operational_trust",
    ):
        assert dimension in report["dimensions"], f"Missing dimension: {dimension}"
    assert "analysis_availability" in report
    assert report.get("limitations")
