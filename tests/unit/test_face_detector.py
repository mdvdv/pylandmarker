import pytest

from landmarker.helpers import load_image
from landmarker.models.face_detector import FaceDetector


@pytest.fixture
def model():
    return FaceDetector()


@pytest.fixture
def image():
    return load_image("assets/face.jpg")


def test_face_detector(model, image):
    detections = model.detect(image)["detections"]
    assert isinstance(detections, list)
    assert len(detections) == 1
