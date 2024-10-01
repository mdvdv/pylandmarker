import mediapipe as mp
import numpy as np

from landmarker.models.base_detector import BaseDetector


class PoseDetector(BaseDetector):
    """MediaPipe Pose Detection implementation.

    PoseDetector processes an RGB image and returns 33 3D body pose landmarks.

    Example:
        >>> from landmarker.models.pose_detector import PoseDetector
        >>> from landmarker.helpers import load_image
        >>> model = PoseDetector()
        >>> image = load_image('assets/pose.jpg')
        >>> output = model.detect(image)
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Args:
            static_image_mode (bool): If set to false, the solution treats
                the input images as a video stream. It will try to detect
                the most prominent person in the very first images, and upon
                a successful detection further localizes the pose landmarks.
                In subsequent images, it then simply tracks those landmarks
                without invoking another detection until it loses track, on
                reducing computation and latency. If set to true, person
                detection runs every input image, ideal for processing a batch
                of static, possibly unrelated, images. Default to true.
            model_complexity (int): Complexity of the pose landmark model: 0,
                1 or 2. Landmark accuracy as well as inference latency
                generally go up with the model complexity. Default to 1.
            smooth_landmarks (bool): If set to true, the solution filters pose
                landmarks across different input images to reduce jitter, but
                ignored if static_image_mode is also set to true.
                Default to true.
            enable_segmentation (bool): If set to true, in addition to the pose
                landmarks the solution also generates the segmentation mask.
                Default to false.
            smooth_segmentation (bool): If set to true, the solution filters
                segmentation masks across different input images to reduce
                jitter. Ignored if enable_segmentation is false or
                static_image_mode is true. Default to true
            min_detection_confidence (float): Minimum confidence value ([0.0,
                1.0]) from the person-detection model for the detection to be
                considered successful. Default to 0.5.
            min_tracking_confidence (float): Minimum confidence value ([0.0,
                1.0]) from the landmark-tracking model for the pose landmarks
                to be considered tracked successfully, or otherwise person
                detection will be invoked automatically on the next input
                image. Setting it to a higher value can increase robustness
                of the solution, at the expense of a higher latency. Ignored
                if static_image_mode is true, where person detection simply
                runs on every image. Default to 0.5.
        """
        super().__init__()

        solution = mp.solutions.pose
        self.pipeline = solution.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> list:
        """
        Args:
            image (np.ndarray): An RGB image represented as NumPy array.

        Returns:
            list: The list with the detected poses.
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
            detection_results["detections"].append(
                {
                    "keypoints_location": [
                        {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                        for landmark in detections.pose_landmarks.landmark
                    ]
                }
            )

        return detection_results
