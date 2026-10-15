import numpy as np

from crop_ai.cv.severity import calculate_severity


def test_severity_uses_lesions_only_inside_leaf() -> None:
    leaf = np.zeros((10, 10), dtype=np.uint8)
    leaf[:5, :] = 255
    lesions = np.zeros_like(leaf)
    lesions[:1, :] = 255
    lesions[8:, :] = 255

    result = calculate_severity(leaf, lesions)

    assert result.affected_area_percent == 20.0
    assert result.category == "Not categorized"
    assert result.status == "experimental"
    assert result.is_experimental is True


def test_severity_uses_only_explicit_operational_thresholds() -> None:
    leaf = np.full((10, 10), 255, dtype=np.uint8)
    lesions = np.zeros_like(leaf)
    lesions[:2, :] = 255

    result = calculate_severity(
        leaf,
        lesions,
        thresholds=((5.0, "Low"), (20.0, "Moderate"), (101.0, "High")),
    )

    assert result.category == "Moderate"
    assert result.status == "experimental"


def test_severity_handles_empty_leaf_mask() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = calculate_severity(mask, mask)
    assert result.affected_area_percent is None
    assert result.category == "Unavailable"
    assert result.status == "unavailable"
