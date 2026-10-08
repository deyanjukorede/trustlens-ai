
"""
TrustLens AI: Unified Assessment Example.

Demonstrates how to load a synthetic CSV dataset, run the
five-dimension TrustLens assessment, and inspect the results.

Run from the repository root:

    python examples/unified_assessment_example.py

This example is for demonstration and learning purposes.
It does not certify a dataset or AI system as trustworthy,
fair, safe, or legally compliant.
"""

from pathlib import Path
import sys

import pandas as pd


# Allow this example to run directly from a source checkout.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trustlens.unified import TrustLensAssessor


def main() -> None:
    """Run and display a unified TrustLens assessment."""

    dataset_path = (
        Path(__file__).resolve().parent
        / "sample_dataset.csv"
    )

    data = pd.read_csv(dataset_path)

    print("=" * 60)
    print("TrustLens AI - Unified Assessment Example")
    print("=" * 60)

    print(f"Dataset: {dataset_path.name}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    assessor = TrustLensAssessor(
        data=data,
        target_column="target",
        prediction_column="prediction",
        sensitive_attributes=["group"],
        identifier_column="record_id",
    )

    report = assessor.assess()

    print("\nAssessment completed.")

    print("\nReport sections:")
    for section in report:
        print(f"  - {section}")

    print("\nAssessment dimensions:")
    dimensions = report.get("dimensions", {})

    for dimension_name in dimensions:
        print(f"  - {dimension_name}")

    print("\nCross-dimension summary:")
    summary = report.get("cross_dimension_summary")

    if summary is None:
        print("  No cross-dimension summary available.")
    else:
        print(summary)

    print("\nImportant:")
    print(
        "TrustLens findings require contextual interpretation "
        "and human review."
    )
    print(
        "This assessment is not a certification of AI safety, "
        "fairness, trustworthiness, or legal compliance."
    )


if __name__ == "__main__":
    main()
