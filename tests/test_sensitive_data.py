import pandas as pd
import pytest

from trustlens.governance.sensitive_data import SensitiveDataDetector


@pytest.fixture
def sensitive_data():
    """Create a representative dataset containing sensitive information."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "full_name": [
                "Ada Smith",
                "John Brown",
                "Mary Jones",
            ],
            "email": [
                "ada@example.com",
                "john@example.com",
                "mary@example.com",
            ],
            "mobile_number": [
                "+447700900001",
                "+447700900002",
                "+447700900003",
            ],
            "age": [28, 35, 42],
        }
    )


def test_detect_by_column_name(sensitive_data):
    """Detector should identify sensitive fields from column names."""
    detector = SensitiveDataDetector(sensitive_data)
    detected = detector.detect_by_column_name()

    assert "full_name" in detected
    assert "email" in detected
    assert "mobile_number" in detected

    assert "name" in detected["full_name"]
    assert "email" in detected["email"]
    assert "phone_number" in detected["mobile_number"]


def test_non_sensitive_column_names_are_not_detected(sensitive_data):
    """Ordinary column names should not be classified as sensitive."""
    detector = SensitiveDataDetector(sensitive_data)
    detected = detector.detect_by_column_name()

    assert "age" not in detected
    assert "customer_id" not in detected


def test_column_name_normalisation():
    """Detector should normalise spaces and hyphens in column names."""
    data = pd.DataFrame(
        {
            "Email Address": ["user@example.com"],
            "Mobile-Number": ["+447700900001"],
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect_by_column_name()

    assert "Email Address" in detected
    assert "Mobile-Number" in detected

    assert "email" in detected["Email Address"]
    assert "phone_number" in detected["Mobile-Number"]


def test_detect_email_by_value_pattern():
    """Email values should be detected even with an unrelated column name."""
    data = pd.DataFrame(
        {
            "contact": [
                "one@example.com",
                "two@example.com",
                "three@example.com",
            ]
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect_by_value_pattern()

    assert "contact" in detected
    assert "email" in detected["contact"]


def test_detect_ip_address_by_value_pattern():
    """IP addresses should be detected from their values."""
    data = pd.DataFrame(
        {
            "network_source": [
                "192.168.1.1",
                "10.0.0.1",
                "172.16.0.1",
            ]
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect_by_value_pattern()

    assert "network_source" in detected
    assert "ip_address" in detected["network_source"]


def test_detect_phone_number_by_value_pattern():
    """Phone numbers should be detected from their values."""
    data = pd.DataFrame(
        {
            "contact_value": [
                "+447700900001",
                "+447700900002",
                "+447700900003",
            ]
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect_by_value_pattern()

    assert "contact_value" in detected
    assert "phone_number" in detected["contact_value"]


def test_threshold_prevents_weak_pattern_detection():
    """A small number of matches should not classify an entire column."""
    data = pd.DataFrame(
        {
            "mixed_values": [
                "user@example.com",
                "ordinary text",
                "another value",
                "not sensitive",
                "general information",
            ]
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect_by_value_pattern(threshold=0.8)

    assert "mixed_values" not in detected


def test_combined_detection_sources():
    """Combined detection should report both applicable detection sources."""
    data = pd.DataFrame(
        {
            "email": [
                "one@example.com",
                "two@example.com",
                "three@example.com",
            ]
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect()

    assert "email" in detected
    assert detected["email"]["sensitive_types"] == ["email"]
    assert detected["email"]["detection_sources"] == [
        "column_name",
        "value_pattern",
    ]


def test_column_name_only_detection_source():
    """Column-name detection should work without matching value patterns."""
    data = pd.DataFrame(
        {
            "full_name": [
                "Ada Smith",
                "John Brown",
                "Mary Jones",
            ]
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect()

    assert "full_name" in detected
    assert "name" in detected["full_name"]["sensitive_types"]
    assert detected["full_name"]["detection_sources"] == ["column_name"]


def test_summary_structure(sensitive_data):
    """Summary should return the expected sensitive-data sections."""
    detector = SensitiveDataDetector(sensitive_data)
    summary = detector.summary()

    assert "sensitive_columns_detected" in summary
    assert "sensitive_columns" in summary
    assert "sensitive_types_detected" in summary
    assert "details" in summary


def test_summary_reports_detected_columns(sensitive_data):
    """Summary should report detected sensitive columns."""
    detector = SensitiveDataDetector(sensitive_data)
    summary = detector.summary()

    assert "full_name" in summary["sensitive_columns"]
    assert "email" in summary["sensitive_columns"]
    assert "mobile_number" in summary["sensitive_columns"]

    assert summary["sensitive_columns_detected"] >= 3


def test_summary_reports_sensitive_types(sensitive_data):
    """Summary should report the sensitive-data types found."""
    detector = SensitiveDataDetector(sensitive_data)
    summary = detector.summary()

    assert "name" in summary["sensitive_types_detected"]
    assert "email" in summary["sensitive_types_detected"]
    assert "phone_number" in summary["sensitive_types_detected"]


def test_invalid_dataframe_input():
    """Detector should reject input that is not a pandas DataFrame."""
    with pytest.raises(TypeError):
        SensitiveDataDetector(["email@example.com"])


def test_empty_dataframe():
    """Detector should reject an empty DataFrame."""
    with pytest.raises(ValueError):
        SensitiveDataDetector(pd.DataFrame())


@pytest.mark.parametrize(
    "sample_size",
    [
        0,
        -1,
        1.5,
        "100",
        None,
    ],
)
def test_invalid_sample_size(sample_size):
    """Detector should reject invalid sample-size values."""
    data = pd.DataFrame({"value": ["example"]})

    with pytest.raises(ValueError):
        SensitiveDataDetector(data, sample_size=sample_size)


def test_invalid_threshold_type():
    """Pattern detection should reject a non-numeric threshold."""
    data = pd.DataFrame({"value": ["example"]})
    detector = SensitiveDataDetector(data)

    with pytest.raises(TypeError):
        detector.detect_by_value_pattern(threshold="high")


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        -0.1,
        1.1,
    ],
)
def test_invalid_threshold_range(threshold):
    """Pattern detection should reject thresholds outside the valid range."""
    data = pd.DataFrame({"value": ["example"]})
    detector = SensitiveDataDetector(data)

    with pytest.raises(ValueError):
        detector.detect_by_value_pattern(threshold=threshold)


def test_plain_text_is_not_detected_as_sensitive():
    """Ordinary text should not be classified as sensitive by value."""
    data = pd.DataFrame(
        {
            "description": [
                "standard customer record",
                "general product information",
                "ordinary business data",
            ]
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect_by_value_pattern()

    assert "description" not in detected


def test_short_numeric_values_are_not_phone_numbers():
    """Short ordinary numeric values should not be classified as phones."""
    data = pd.DataFrame(
        {
            "age": [25, 31, 40],
            "score": [100, 200, 300],
        }
    )

    detector = SensitiveDataDetector(data)
    detected = detector.detect_by_value_pattern()

    assert "age" not in detected
    assert "score" not in detected


def test_sample_size_limits_values_inspected():
    """Detector should respect the configured value sample size."""
    data = pd.DataFrame(
        {
            "contact": [
                "one@example.com",
                "two@example.com",
                "ordinary text",
                "another ordinary value",
            ]
        }
    )

    detector = SensitiveDataDetector(data, sample_size=2)
    detected = detector.detect_by_value_pattern()

    assert "contact" in detected
    assert "email" in detected["contact"]
