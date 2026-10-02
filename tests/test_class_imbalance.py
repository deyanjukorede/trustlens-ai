import pandas as pd
import pytest

from trustlens.readiness.class_imbalance import ClassImbalanceAnalyzer


@pytest.fixture
def balanced_data():
    """Create a balanced binary classification dataset."""
    return pd.DataFrame(
        {
            "age": [25, 31, 42, 36, 29, 48],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def imbalanced_data():
    """Create an imbalanced classification dataset."""
    return pd.DataFrame(
        {
            "feature": list(range(10)),
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )


def test_class_counts(balanced_data):
    """Analyzer should correctly count target classes."""
    analyzer = ClassImbalanceAnalyzer(
        balanced_data,
        target_column="target",
    )

    assert analyzer.class_counts() == {0: 3, 1: 3}


def test_class_proportions(balanced_data):
    """Analyzer should correctly calculate class proportions."""
    analyzer = ClassImbalanceAnalyzer(
        balanced_data,
        target_column="target",
    )

    proportions = analyzer.class_proportions()

    assert proportions[0] == 50.0
    assert proportions[1] == 50.0


def test_balanced_dataset_not_flagged(balanced_data):
    """Balanced target distribution should not be flagged."""
    analyzer = ClassImbalanceAnalyzer(
        balanced_data,
        target_column="target",
    )

    assert analyzer.is_imbalanced() is False
    assert analyzer.minority_classes() == []


def test_imbalanced_dataset_is_flagged(imbalanced_data):
    """Underrepresented target classes should be detected."""
    analyzer = ClassImbalanceAnalyzer(
        imbalanced_data,
        target_column="target",
    )

    assert analyzer.is_imbalanced() is True
    assert analyzer.minority_classes() == [1]


def test_majority_class(imbalanced_data):
    """Analyzer should identify the majority class."""
    analyzer = ClassImbalanceAnalyzer(
        imbalanced_data,
        target_column="target",
    )

    assert analyzer.majority_class() == 0


def test_minority_class(imbalanced_data):
    """Analyzer should identify the minority class."""
    analyzer = ClassImbalanceAnalyzer(
        imbalanced_data,
        target_column="target",
    )

    assert analyzer.minority_class() == 1


def test_imbalance_ratio_balanced(balanced_data):
    """Balanced classes should have an imbalance ratio of one."""
    analyzer = ClassImbalanceAnalyzer(
        balanced_data,
        target_column="target",
    )

    assert analyzer.imbalance_ratio() == 1.0


def test_imbalance_ratio_imbalanced(imbalanced_data):
    """Analyzer should calculate majority-to-minority ratio."""
    analyzer = ClassImbalanceAnalyzer(
        imbalanced_data,
        target_column="target",
    )

    assert analyzer.imbalance_ratio() == 9.0


def test_custom_threshold():
    """Analyzer should respect a custom imbalance threshold."""
    data = pd.DataFrame(
        {
            "target": [0, 0, 0, 1],
        }
    )

    analyzer = ClassImbalanceAnalyzer(
        data,
        target_column="target",
        imbalance_threshold=0.30,
    )

    assert analyzer.is_imbalanced() is True
    assert analyzer.minority_classes() == [1]


def test_missing_target_summary():
    """Analyzer should report missing target values."""
    data = pd.DataFrame(
        {
            "target": [0, 1, None, 0],
        }
    )

    analyzer = ClassImbalanceAnalyzer(
        data,
        target_column="target",
    )

    summary = analyzer.missing_target_summary()

    assert summary["missing_count"] == 1
    assert summary["missing_rate"] == 25.0


def test_assess_returns_expected_sections(imbalanced_data):
    """Full assessment should expose class imbalance information."""
    analyzer = ClassImbalanceAnalyzer(
        imbalanced_data,
        target_column="target",
    )

    report = analyzer.assess()

    assert report["target_column"] == "target"
    assert report["class_count"] == 2
    assert "class_counts" in report
    assert "class_proportions" in report
    assert "majority_class" in report
    assert "minority_class" in report
    assert "minority_classes" in report
    assert "imbalance_ratio" in report
    assert "imbalance_threshold" in report
    assert "is_imbalanced" in report
    assert "missing_target" in report


def test_assess_imbalanced_result(imbalanced_data):
    """Full assessment should report the expected imbalance."""
    analyzer = ClassImbalanceAnalyzer(
        imbalanced_data,
        target_column="target",
    )

    report = analyzer.assess()

    assert report["class_counts"] == {0: 9, 1: 1}
    assert report["class_proportions"] == {0: 90.0, 1: 10.0}
    assert report["majority_class"] == 0
    assert report["minority_class"] == 1
    assert report["minority_classes"] == [1]
    assert report["imbalance_ratio"] == 9.0
    assert report["is_imbalanced"] is True


def test_multiclass_imbalance():
    """Analyzer should support multiclass target variables."""
    data = pd.DataFrame(
        {
            "target": [
                "A", "A", "A", "A", "A",
                "B", "B", "B", "B",
                "C",
            ],
        }
    )

    analyzer = ClassImbalanceAnalyzer(
        data,
        target_column="target",
    )

    report = analyzer.assess()

    assert report["class_count"] == 3
    assert report["majority_class"] == "A"
    assert report["minority_class"] == "C"
    assert report["minority_classes"] == ["C"]
    assert report["is_imbalanced"] is True


def test_invalid_input():
    """Analyzer should reject non-DataFrame input."""
    with pytest.raises(TypeError):
        ClassImbalanceAnalyzer(
            [0, 1, 0, 1],
            target_column="target",
        )


def test_empty_dataframe():
    """Analyzer should reject an empty DataFrame."""
    with pytest.raises(ValueError):
        ClassImbalanceAnalyzer(
            pd.DataFrame(),
            target_column="target",
        )


def test_invalid_target_column(balanced_data):
    """Analyzer should reject a target column that does not exist."""
    with pytest.raises(ValueError):
        ClassImbalanceAnalyzer(
            balanced_data,
            target_column="missing_target",
        )


@pytest.mark.parametrize(
    "threshold",
    [0, -0.1, 1.1],
)
def test_invalid_imbalance_threshold(balanced_data, threshold):
    """Analyzer should reject invalid imbalance thresholds."""
    with pytest.raises(ValueError):
        ClassImbalanceAnalyzer(
            balanced_data,
            target_column="target",
            imbalance_threshold=threshold,
        )
