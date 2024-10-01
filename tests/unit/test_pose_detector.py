import pytest

from landmarker.helpers import load_image
from landmarker.models.pose_detector import PoseDetector


@pytest.fixture
def model():
    return PoseDetector()


@pytest.fixture
def image():
    return load_image("assets/pose.jpg")


def test_pose_detector(model, image):
    detections = model.detect(image)["detections"]
    assert isinstance(detections, list)
    assert len(detections) == 1
