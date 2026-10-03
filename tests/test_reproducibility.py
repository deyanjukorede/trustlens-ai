import pandas as pd
import pytest

from trustlens.operational_trust.reproducibility import (
    ReproducibilityReadinessAnalyzer,
)


def test_analyzer_initialises_with_valid_dataframe():
    """Analyzer should initialise with valid data."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    analyzer = ReproducibilityReadinessAnalyzer(data)

    assert analyzer.row_count == 3
    assert analyzer.column_count == 1
    assert analyzer.metadata == {}
    assert analyzer.identifier_column is None


def test_analyzer_rejects_non_dataframe_input():
    """Data must be supplied as a pandas DataFrame."""
    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
        ReproducibilityReadinessAnalyzer(
            {
                "feature": [1, 2, 3],
            }
        )


def test_analyzer_rejects_empty_dataframe():
    """Data must not be empty."""
    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        ReproducibilityReadinessAnalyzer(pd.DataFrame())


def test_analyzer_rejects_non_dictionary_metadata():
    """Metadata must be a dictionary when supplied."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary when supplied",
    ):
        ReproducibilityReadinessAnalyzer(
            data,
            metadata=["dataset_version"],
        )


def test_identifier_column_must_be_string():
    """Identifier column should require a string when supplied."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
        }
    )

    with pytest.raises(
        TypeError,
        match="identifier_column must be a string",
    ):
        ReproducibilityReadinessAnalyzer(
            data,
            identifier_column=123,
        )


def test_identifier_column_must_not_be_empty():
    """Blank identifier names should be rejected."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="identifier_column must not be empty",
    ):
        ReproducibilityReadinessAnalyzer(
            data,
            identifier_column="   ",
        )


def test_identifier_column_must_exist_in_data():
    """Specified identifier should exist in the dataset."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="identifier_column must exist in data",
    ):
        ReproducibilityReadinessAnalyzer(
            data,
            identifier_column="unknown_id",
        )


def test_missing_metadata_is_reported():
    """Absent reproducibility evidence should be explicitly reported."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    result = ReproducibilityReadinessAnalyzer(data).assess()

    assert result["metadata_supplied"] is False

    assert result["missing_metadata_fields"] == [
        "dataset_version",
        "code_version",
        "random_seed",
        "environment",
        "data_source",
        "lineage",
        "execution_id",
    ]

    assert result["metadata_coverage"] == {
        "recognised_fields": 7,
        "provided_fields": 0,
        "missing_fields": 7,
        "coverage_rate": 0.0,
    }

    assert result["review_required"] is True

    assert result["review_reasons"] == [
        "missing_reproducibility_metadata"
    ]


def test_complete_metadata_produces_full_evidence_coverage():
    """All recognised evidence should produce complete coverage."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "dataset_version": "dataset-v1",
        "code_version": "abc123",
        "random_seed": 42,
        "environment": "python-3.11",
        "data_source": "warehouse",
        "lineage": "raw -> curated -> model",
        "execution_id": "run-001",
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert result["metadata_supplied"] is True

    assert result["missing_metadata_fields"] == []

    assert result["metadata_coverage"] == {
        "recognised_fields": 7,
        "provided_fields": 7,
        "missing_fields": 0,
        "coverage_rate": 1.0,
    }

    assert result["review_required"] is False
    assert result["review_reasons"] == []


def test_partial_metadata_reports_missing_evidence():
    """Partial metadata should preserve supplied and missing evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "dataset_version": "v2",
        "code_version": "commit-123",
        "random_seed": 7,
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert result["metadata_coverage"] == {
        "recognised_fields": 7,
        "provided_fields": 3,
        "missing_fields": 4,
        "coverage_rate": 0.4286,
    }

    assert result["missing_metadata_fields"] == [
        "environment",
        "data_source",
        "lineage",
        "execution_id",
    ]

    assert result["review_required"] is True


def test_blank_string_metadata_does_not_count_as_evidence():
    """Blank strings should be treated as missing evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "dataset_version": "   ",
        "code_version": "",
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert (
        result["metadata_evidence"]["dataset_version"]["provided"]
        is False
    )

    assert (
        result["metadata_evidence"]["code_version"]["provided"]
        is False
    )

    assert "dataset_version" in result["missing_metadata_fields"]
    assert "code_version" in result["missing_metadata_fields"]


def test_random_seed_zero_counts_as_valid_evidence():
    """Random seed zero should not be mistaken for missing evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "random_seed": 0,
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert (
        result["metadata_evidence"]["random_seed"]["provided"]
        is True
    )

    assert (
        result["metadata_evidence"]["random_seed"]["value"]
        == 0
    )

    assert "random_seed" not in result["missing_metadata_fields"]


def test_boolean_metadata_counts_as_supplied_evidence():
    """Boolean values should remain valid supplied metadata."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "environment": False,
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert (
        result["metadata_evidence"]["environment"]["provided"]
        is True
    )


def test_empty_collection_metadata_does_not_count_as_evidence():
    """Empty collections should be treated as absent evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "lineage": [],
        "environment": {},
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert result["metadata_evidence"]["lineage"]["provided"] is False
    assert (
        result["metadata_evidence"]["environment"]["provided"]
        is False
    )


def test_additional_metadata_fields_are_preserved():
    """Unrecognised metadata should be surfaced without rejection."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    metadata = {
        "dataset_version": "v1",
        "model_version": "model-v4",
        "owner": "data-team",
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
    ).assess()

    assert result["additional_metadata_fields"] == [
        "model_version",
        "owner",
    ]


def test_identifier_analysis_when_identifier_not_supplied():
    """No identifier should be treated as unavailable evidence."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    result = ReproducibilityReadinessAnalyzer(data).assess()

    assert result["identifier_analysis"] == {
        "provided": False,
        "column": None,
        "missing_values": None,
        "duplicate_values": None,
        "unique_values": None,
        "is_complete": None,
        "is_unique": None,
        "requires_review": False,
    }


def test_complete_unique_identifier_does_not_require_review():
    """Complete unique identifiers should pass integrity checks."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3", "R4"],
            "feature": [10, 20, 30, 40],
        }
    )

    result = ReproducibilityReadinessAnalyzer(
        data,
        identifier_column="record_id",
    ).assess()

    identifier = result["identifier_analysis"]

    assert identifier["provided"] is True
    assert identifier["column"] == "record_id"
    assert identifier["missing_values"] == 0
    assert identifier["duplicate_values"] == 0
    assert identifier["unique_values"] == 4
    assert identifier["is_complete"] is True
    assert identifier["is_unique"] is True
    assert identifier["requires_review"] is False


def test_identifier_with_missing_values_requires_review():
    """Missing identifier values should be surfaced for review."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", None, "R3", "R4"],
            "feature": [10, 20, 30, 40],
        }
    )

    metadata = {
        "dataset_version": "v1",
        "code_version": "commit-1",
        "random_seed": 42,
        "environment": "python-3.11",
        "data_source": "warehouse",
        "lineage": "source -> dataset",
        "execution_id": "run-001",
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
        identifier_column="record_id",
    ).assess()

    identifier = result["identifier_analysis"]

    assert identifier["missing_values"] == 1
    assert identifier["is_complete"] is False
    assert identifier["requires_review"] is True

    assert result["review_reasons"] == [
        "identifier_integrity"
    ]


def test_duplicate_identifier_values_require_review():
    """Duplicate identifiers should be surfaced for review."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", "R1", "R2", "R3"],
            "feature": [10, 20, 30, 40],
        }
    )

    metadata = {
        "dataset_version": "v1",
        "code_version": "commit-1",
        "random_seed": 42,
        "environment": "python-3.11",
        "data_source": "warehouse",
        "lineage": "source -> dataset",
        "execution_id": "run-001",
    }

    result = ReproducibilityReadinessAnalyzer(
        data,
        metadata=metadata,
        identifier_column="record_id",
    ).assess()

    identifier = result["identifier_analysis"]

    assert identifier["duplicate_values"] == 1
    assert identifier["is_unique"] is False
    assert identifier["requires_review"] is True

    assert result["review_reasons"] == [
        "identifier_integrity"
    ]


def test_missing_metadata_and_identifier_issue_can_coexist():
    """Independent reproducibility indicators should coexist."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", "R1", None],
            "feature": [10, 20, 30],
        }
    )

    result = ReproducibilityReadinessAnalyzer(
        data,
        identifier_column="record_id",
    ).assess()

    assert result["review_required"] is True

    assert result["review_reasons"] == [
        "missing_reproducibility_metadata",
        "identifier_integrity",
    ]


def test_assessment_returns_expected_top_level_structure():
    """Assessment should expose a stable output contract."""
    data = pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3"],
            "feature": [10, 20, 30],
        }
    )

    result = ReproducibilityReadinessAnalyzer(
        data,
        identifier_column="record_id",
    ).assess()

    assert set(result.keys()) == {
        "dataset",
        "metadata_supplied",
        "metadata_evidence",
        "metadata_coverage",
        "missing_metadata_fields",
        "additional_metadata_fields",
        "identifier_analysis",
        "review_required",
        "review_reasons",
    }
