import mediapipe as mp
import numpy as np

from landmarker.models.base_detector import BaseDetector


class FaceDetector(BaseDetector):
    """MediaPipe Face Detection implementation.

    FaceDetector processes an RGB image and detects face with 6 landmarks and
    multi-face support.

    Example:
        >>> from landmarker.models.face_detector import FaceDetector
        >>> from landmarker.helpers import load_image
        >>> model = FaceDetector()
        >>> image = load_image('assets/face.jpg')
        >>> output = model.detect(image)
    """

    def __init__(
        self, model_selection: int = 0, min_detection_confidence: float = 0.5
    ) -> None:
        """
        Args:
            model_selection (int): An integer index 0 or 1. Use 0 to select a
                short-range model that works best for faces within 2 meters
                from the camera, and 1 for a full-range model best for faces
                within 5 meters. For the full-range option, a sparse model is
                used for its improved inference speed. Default to 0.
            min_detection_confidence (float): Minimum confidence value ([0.0,
                1.0]) from the face detection model for the detection to be
                considered successful. Default to 0.5.
        """
        super().__init__()

        solution = mp.solutions.face_detection
        self.pipeline = solution.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, image: np.ndarray) -> list:
        """
        Args:
            image (np.ndarray): An RGB image represented as NumPy array.

        Returns:
            list: The list with the detected faces.
        """
        detections = self.pipeline.process(image)
        detection_results = {
            "meta": {
                "image_width": image.shape[1],
                "image_height": image.shape[0],
            },
            "detections": [],
        }

        if detections:
            for detection in detections.detections:
                locations = detection.location_data
                detection_results["detections"].append(
                    {
                        "label_id": detection.label_id[0],
                        "score": detection.score[0],
                        "bounding_box_location": {
                            "xmin": locations.relative_bounding_box.xmin,
                            "ymin": locations.relative_bounding_box.ymin,
                            "width": locations.relative_bounding_box.width,
                            "height": locations.relative_bounding_box.height,
                        },
                        "keypoints_location": [
                            {"x": keypoint.x, "y": keypoint.y}
                            for keypoint in locations.relative_keypoints
                        ],
                    }
                )

        return detection_results
