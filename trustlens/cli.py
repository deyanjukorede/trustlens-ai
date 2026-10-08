
"""
Command-line interface for TrustLens AI.

Run unified assessments on CSV datasets and export structured
JSON reports.

Example
-------
python -m trustlens.cli assess dataset.csv --output report.json

The CLI does not certify that a dataset or AI system is safe,
fair, trustworthy, production-ready, or legally compliant.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .unified import TrustLensAssessor


def build_parser() -> argparse.ArgumentParser:
    """Create the TrustLens command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="trustlens",
        description=(
            "TrustLens AI: evidence-based data readiness, "
            "governance, fairness, and operational trust assessment."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    assess_parser = subparsers.add_parser(
        "assess",
        help="Assess a CSV dataset and export a JSON report.",
    )

    assess_parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the CSV dataset to assess.",
    )

    assess_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the JSON report to create.",
    )

    assess_parser.add_argument(
        "--target",
        help="Optional prediction target column.",
    )

    assess_parser.add_argument(
        "--prediction",
        help="Optional model prediction column.",
    )

    assess_parser.add_argument(
        "--sensitive",
        nargs="+",
        metavar="COLUMN",
        help="Optional sensitive-attribute column names.",
    )

    assess_parser.add_argument(
        "--reference",
        type=Path,
        help="Optional reference CSV for drift analysis.",
    )

    assess_parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional governance metadata JSON file.",
    )

    assess_parser.add_argument(
        "--governance-controls",
        type=Path,
        help="Optional governance controls JSON file.",
    )

    assess_parser.add_argument(
        "--reproducibility-metadata",
        type=Path,
        help="Optional reproducibility metadata JSON file.",
    )

    assess_parser.add_argument(
        "--monitoring-metadata",
        type=Path,
        help="Optional monitoring metadata JSON file.",
    )

    assess_parser.add_argument(
        "--identifier",
        help="Optional identifier column.",
    )

    return parser


def load_csv(path: Path) -> pd.DataFrame:
    """Load a non-empty CSV dataset."""
    if not path.is_file():
        raise ValueError(
            f"CSV file does not exist or is not a file: {path}"
        )

    try:
        data = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Unable to read CSV file '{path}': {exc}"
        ) from exc

    if data.empty:
        raise ValueError(
            f"CSV dataset must not be empty: {path}"
        )

    return data


def load_json_object(
    path: Optional[Path],
    label: str,
) -> Optional[Dict[str, Any]]:
    """Load optional JSON metadata, requiring a JSON object."""
    if path is None:
        return None

    if not path.is_file():
        raise ValueError(
            f"{label} file does not exist or is not a file: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read {label} JSON file '{path}': {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            f"{label} must contain a JSON object."
        )

    return result


def json_safe(value: Any) -> Any:
    """
    Convert common scientific Python values to JSON-compatible types.

    Non-finite floating-point values are represented as null.
    Unsupported object types raise TypeError rather than being
    silently converted to potentially sensitive string values.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat() if not pd.isna(value) else None

    if isinstance(value, pd.Timedelta):
        return value.isoformat() if not pd.isna(value) else None

    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())

    if isinstance(value, pd.Index):
        return json_safe(value.tolist())

    if hasattr(value, "item") and callable(value.item):
        return json_safe(value.item())

    if value is pd.NA or value is pd.NaT:
        return None

    raise TypeError(
        f"Unsupported JSON report value type: "
        f"{type(value).__name__}"
    )


def write_json_report(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write a structured JSON report to the requested location."""
    if output_path.exists():
        raise ValueError(
            f"Output already exists; refusing to overwrite: "
            f"{output_path}"
        )

    if not output_path.parent.is_dir():
        raise ValueError(
            f"Output directory does not exist: "
            f"{output_path.parent}"
        )

    safe_report = json_safe(report)

    try:
        serialized = json.dumps(
            safe_report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

        with output_path.open("x", encoding="utf-8") as file:
            file.write(serialized)
            file.write("\n")

    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to write JSON report '{output_path}': {exc}"
        ) from exc


def run_assessment(args: argparse.Namespace) -> Path:
    """Execute the requested TrustLens assessment."""
    data = load_csv(args.dataset)

    reference_data = (
        load_csv(args.reference)
        if args.reference is not None
        else None
    )

    metadata = load_json_object(
        args.metadata,
        "Governance metadata",
    )

    governance_controls = load_json_object(
        args.governance_controls,
        "Governance controls",
    )

    reproducibility_metadata = load_json_object(
        args.reproducibility_metadata,
        "Reproducibility metadata",
    )

    monitoring_metadata = load_json_object(
        args.monitoring_metadata,
        "Monitoring metadata",
    )

    assessor = TrustLensAssessor(
        data=data,
        target_column=args.target,
        prediction_column=args.prediction,
        sensitive_attributes=args.sensitive,
        metadata=metadata,
        governance_controls=governance_controls,
        reference_data=reference_data,
        reproducibility_metadata=reproducibility_metadata,
        identifier_column=args.identifier,
        monitoring_metadata=monitoring_metadata,
    )

    report = assessor.assess()

    write_json_report(report, args.output)

    return args.output


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "assess":
            output_path = run_assessment(args)
            print(
                f"TrustLens assessment completed. "
                f"Report saved to: {output_path}"
            )
            return 0

        parser.error("Unsupported command.")

    except (ValueError, TypeError, KeyError) as exc:
        print(
            f"TrustLens error: {exc}",
            file=sys.stderr,
        )
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
