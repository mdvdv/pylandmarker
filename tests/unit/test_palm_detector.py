import pytest

from landmarker.helpers import load_image
from landmarker.models.palm_detector import PalmDetector


@pytest.fixture
def model():
    return PalmDetector()


@pytest.fixture
def image():
    return load_image("assets/palms.jpg")


def test_palm_detector(model, image):
    detections = model.detect(image)["detections"]
    assert isinstance(detections, list)
    assert len(detections) == 2
