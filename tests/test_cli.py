
"""
Automated tests for the TrustLens AI command-line interface.

Covers CSV loading, JSON metadata loading, report export,
command-line execution, error handling, and overwrite protection.
"""

import json
import subprocess
import sys

import pandas as pd
import pytest

from trustlens.cli import (
    json_safe,
    load_csv,
    load_json_object,
    main,
    write_json_report,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a representative CSV dataset for CLI testing."""
    path = tmp_path / "dataset.csv"

    data = pd.DataFrame(
        {
            "age": [22, 34, 45, 29, 51, 38, 27, 43, 36, 49],
            "income": [
                25000, 42000, 58000, 31000, 67000,
                48000, 29000, 55000, 44000, 63000,
            ],
            "group": [
                "A", "B", "A", "B", "A",
                "B", "A", "B", "A", "B",
            ],
            "target": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
            "prediction": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        }
    )

    data.to_csv(path, index=False)

    return path


def test_load_csv_successfully(sample_csv):
    """A valid CSV should load as a pandas DataFrame."""
    data = load_csv(sample_csv)

    assert isinstance(data, pd.DataFrame)
    assert data.shape == (10, 5)


def test_load_csv_rejects_missing_file(tmp_path):
    """Missing CSV files should produce a clear error."""
    missing = tmp_path / "missing.csv"

    with pytest.raises(ValueError, match="does not exist"):
        load_csv(missing)


def test_load_csv_rejects_empty_dataset(tmp_path):
    """CSV files containing headers only should be rejected."""
    path = tmp_path / "empty.csv"
    path.write_text("age,income\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        load_csv(path)


def test_load_json_object_successfully(tmp_path):
    """Valid JSON metadata objects should load correctly."""
    path = tmp_path / "metadata.json"

    path.write_text(
        json.dumps({"dataset_owner": "Data Team"}),
        encoding="utf-8",
    )

    result = load_json_object(path, "Governance metadata")

    assert result == {"dataset_owner": "Data Team"}


def test_load_json_object_accepts_missing_optional_input():
    """Optional JSON metadata may be omitted."""
    assert load_json_object(None, "Metadata") is None


def test_load_json_object_rejects_invalid_json(tmp_path):
    """Malformed JSON must be rejected."""
    path = tmp_path / "invalid.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read"):
        load_json_object(path, "Metadata")


def test_load_json_object_rejects_non_object(tmp_path):
    """JSON arrays must not be accepted as metadata objects."""
    path = tmp_path / "array.json"
    path.write_text('["one", "two"]', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_json_object(path, "Metadata")


def test_json_safe_converts_non_finite_values():
    """NaN and infinity must become JSON null values."""
    result = json_safe(
        {
            "nan": float("nan"),
            "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
        }
    )

    assert result == {
        "nan": None,
        "positive_infinity": None,
        "negative_infinity": None,
    }


def test_json_safe_converts_pandas_timestamp():
    """Pandas timestamps should become ISO-formatted strings."""
    result = json_safe(
        {"created_at": pd.Timestamp("2026-01-15")}
    )

    assert result["created_at"] == "2026-01-15T00:00:00"


def test_json_safe_rejects_unsupported_objects():
    """Unsupported types must not be silently stringified."""
    with pytest.raises(TypeError, match="Unsupported JSON"):
        json_safe(object())


def test_write_json_report_successfully(tmp_path):
    """JSON reports should be written to the requested path."""
    output = tmp_path / "report.json"

    write_json_report(
        {"framework": "TrustLens AI", "score": 95.5},
        output,
    )

    assert output.is_file()

    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["framework"] == "TrustLens AI"
    assert saved["score"] == 95.5


def test_write_json_report_refuses_overwrite(tmp_path):
    """An existing report must never be overwritten."""
    output = tmp_path / "report.json"
    output.write_text("original content", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_json_report({"test": True}, output)

    assert output.read_text(encoding="utf-8") == "original content"


def test_write_json_report_rejects_missing_directory(tmp_path):
    """Report output directories must already exist."""
    output = tmp_path / "missing_directory" / "report.json"

    with pytest.raises(ValueError, match="does not exist"):
        write_json_report({"test": True}, output)


def test_cli_generates_unified_json_report(sample_csv, tmp_path):
    """The CLI should generate a complete unified JSON report."""
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "assess",
            str(sample_csv),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.is_file()

    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["framework"] == "TrustLens AI"
    assert report["report_type"] == "unified_assessment"
    assert len(report["dimensions"]) == 5
    assert "cross_dimension_summary" in report


def test_cli_accepts_modelling_context(sample_csv, tmp_path):
    """Target, prediction and sensitive columns should propagate."""
    output = tmp_path / "model_report.json"

    exit_code = main(
        [
            "assess",
            str(sample_csv),
            "--target",
            "target",
            "--prediction",
            "prediction",
            "--sensitive",
            "group",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    context = report["assessment_context"]

    assert context["target_column"] == "target"
    assert context["prediction_column"] == "prediction"
    assert context["sensitive_attributes"] == ["group"]


def test_cli_accepts_reference_dataset(sample_csv, tmp_path):
    """Reference CSV input should enable drift analysis."""
    output = tmp_path / "reference_report.json"

    exit_code = main(
        [
            "assess",
            str(sample_csv),
            "--reference",
            str(sample_csv),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0

    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["assessment_context"]["reference_data_supplied"] is True
    assert report["dimensions"]["operational_trust"]["data_drift"] is not None


def test_cli_reports_missing_dataset_error(tmp_path, capsys):
    """Missing datasets should produce a nonzero exit code."""
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "assess",
            str(tmp_path / "missing.csv"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert "TrustLens error:" in capsys.readouterr().err
    assert not output.exists()


def test_cli_prevents_overwriting_report(sample_csv, tmp_path, capsys):
    """Existing report files must remain unchanged."""
    output = tmp_path / "report.json"
    output.write_text("original", encoding="utf-8")

    exit_code = main(
        [
            "assess",
            str(sample_csv),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "original"


def test_cli_rejects_unknown_target(sample_csv, tmp_path, capsys):
    """Unknown target columns should return an error."""
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "assess",
            str(sample_csv),
            "--target",
            "unknown_target",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert "target_column" in capsys.readouterr().err
    assert not output.exists()


def test_cli_module_help_command():
    """The CLI module should be executable with Python -m."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trustlens.cli",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "assess" in result.stdout
    assert "TrustLens" in result.stdout


def test_cli_module_assess_help_command():
    """The assess subcommand should document its options."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trustlens.cli",
            "assess",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--output" in result.stdout
    assert "--target" in result.stdout
    assert "--reference" in result.stdout
