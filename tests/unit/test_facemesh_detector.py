import pytest

from landmarker.helpers import load_image
from landmarker.models.facemesh_detector import FaceMeshDetector


@pytest.fixture
def model():
    return FaceMeshDetector()


@pytest.fixture
def image():
    return load_image("assets/face.jpg")


def test_face_mesh_detector(model, image):
    detections = model.detect(image)["detections"]
    assert isinstance(detections, list)
    assert len(detections) == 1
