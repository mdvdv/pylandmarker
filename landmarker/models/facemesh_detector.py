import mediapipe as mp
import numpy as np

from landmarker.models.base_detector import BaseDetector


class FaceMeshDetector(BaseDetector):
    """MediaPipe Face Mesh Detection implementation.

    FaceMeshDetector processes an RGB image and estimates 468 3D face landmarks
    for each detected face.

    Example:
        >>> from landmarker.models.facemesh_detector import FaceMeshDetector
        >>> from landmarker.helpers import load_image
        >>> model = FaceMeshDetector()
        >>> image = load_image('assets/face.jpg')
        >>> output = model.detect(image)
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Args:
            static_image_mode (bool): If set to false, the solution treats the
                input images as a video stream. It will try to detect faces in
                the first input images, and upon a successful detection further
                localizes the face landmarks. In subsequent images, once all
                max_num_faces faces are detected and the corresponding face
                landmarks are localized, it simply tracks those landmarks
                without invoking another detection until it loses track of any
                of the faces. This reduces latency and is ideal for processing
                video frames. If set to true, face detection runs on every
                input image, ideal for processing a batch of static, possibly
                unrelated, images. Default to true.
            max_num_faces (int): Maximum number of faces to detect.
                Default to 1.
            refine_landmarks (bool): Whether to further refine the landmark
                coordinates around the eyes and lips, and output additional
                landmarks around the irises by applying the Attention Mesh
                Model. Default to true.
            min_detection_confidence (float): Minimum confidence value ([0.0,
                1.0]) from the face detection model for the detection to be
                considered successful. Default to 0.5.
            min_tracking_confidence (float): Minimum confidence value ([0.0,
                1.0]) from the landmark-tracking model for the face landmarks
                to be considered tracked successfully, or otherwise face
                detection will be invoked automatically on the next input
                image. Setting it to a higher value can increase robustness
                of the solution, at the expense of a higher latency. Ignored
                if static_image_mode is true, where face detection simply
                runs on every image. Default to 0.5.
        """
        super().__init__()

        solution = mp.solutions.face_mesh
        self.pipeline = solution.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> list:
        """
        Args:
            image (np.ndarray): An RGB image represented as NumPy array.

        Returns:
            list: The list with the detected face meshes.
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
            for detection in detections.multi_face_landmarks:
                detection_results["detections"].append(
                    {
                        "keypoints_location": [
                            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                            for landmark in detection.landmark
                        ]
                    }
                )

        return detection_results
