import mediapipe as mp
import numpy as np

from landmarker.models.base_detector import BaseDetector


class PalmDetector(BaseDetector):
    """MediaPipe Hands Detection implementation.

    PalmDetector processes an RGB image and returns 21 hand landmarks and
    handedness of each detected hand.

    Example:
        >>> from landmarker.models.palm_detector import PalmDetector
        >>> from landmarker.helpers import load_image
        >>> model = PalmDetector()
        >>> image = load_image('assets/palms.jpg')
        >>> output = model.detect(image)
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Args:
            static_image_mode (bool): If set to false, the solution treats the
                input images as a video stream.It will try to detect hands in
                the first input images, and upon a successful detection further
                localizes the hand landmarks. In subsequent images, once all
                max_num_hands hands are detected and the corresponding hand
                landmarks are localized, it simply tracks those landmarks
                without invoking another detection until it loses track of any
                of the hands. This reduces latency and is ideal for processing
                video frames. If set to true, hand detection runs on every
                input image, ideal for processing a batch of static, possibly
                unrelated, images. Default to true.
            max_num_hands (int): Maximum number of hands to detect.
                Default to 2.
            model_complexity (int): Complexity of the hand landmark model: 0 or
                1. Landmark accuracy as well as inference latency generally go
                up with the model complexity. Default to 1.
            min_detection_confidence (float): Minimum confidence value ([0.0,
                1.0]) from the hand detection model for the detection to be
                considered successful. Default to 0.5.
            min_tracking_confidence (float): Minimum confidence value ([0.0,
                1.0]) from the landmark-tracking model for the hand landmarks
                to be considered tracked successfully, or otherwise hand
                detection will be invoked automatically on the next input
                image. Setting it to a higher value can increase robustness
                of the solution, at the expense of a higher latency. Ignored
                if static_image_mode is true, where hand detection simply runs
                on every image. Default to 0.5.
        """
        super().__init__()

        solution = mp.solutions.hands
        self.pipeline = solution.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> list:
        """
        Args:
            image (np.ndarray): An RGB image represented as NumPy array.

        Returns:
            list: The list with the detected hands.
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
            for location, handedness in zip(
                detections.multi_hand_landmarks, detections.multi_handedness
            ):
                detection_results["detections"].append(
                    {
                        "handedness": [
                            {
                                "index": hand.index,
                                "score": hand.score,
                                "label": hand.label,
                            }
                            for hand in handedness.classification
                        ],
                        "keypoints_location": [
                            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                            for landmark in location.landmark
                        ],
                    }
                )

        return detection_results
