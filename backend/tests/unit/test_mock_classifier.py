import numpy as np

from crop_ai.models.mock_classifier import MockClassifier


def test_mock_classifier_is_deterministic_and_explicit() -> None:
    image = np.full((96, 96, 3), (10, 120, 30), dtype=np.uint8)
    classifier = MockClassifier()
    first = classifier.predict(image)
    second = classifier.predict(image)

    assert first == second
    assert first.is_mock is True
    assert len(first.top_k) == 3
    assert sum(item.confidence for item in first.top_k) == 1.0
