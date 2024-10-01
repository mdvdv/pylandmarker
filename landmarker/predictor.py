from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np

from landmarker.models.base_detector import BaseDetector
from landmarker.models.face_detector import FaceDetector
from landmarker.models.facemesh_detector import FaceMeshDetector
from landmarker.models.palm_detector import PalmDetector
from landmarker.models.pose_detector import PoseDetector

ACCEPTED_DETECTION_TASKS = ["face", "facemesh", "palm", "pose"]


@dataclass
class Predictor(ABC):
    def __init__(self, task: str, config: dict):
        self.model = build_model(task=task, config=config)
        super().__init__()

    def predict(self, image: np.ndarray):
        predictions = self.model.detect(image)
        return predictions


def build_model(task: str, config: dict) -> BaseDetector:
    if task not in ACCEPTED_DETECTION_TASKS:
        raise ValueError(f"task must be one of {ACCEPTED_DETECTION_TASKS}.")

    if task == "face":
        return FaceDetector(**config["model"])
    elif task == "facemesh":
        return FaceMeshDetector(**config["model"])
    elif task == "palm":
        return PalmDetector(**config["model"])
    elif task == "pose":
        return PoseDetector(**config["model"])
