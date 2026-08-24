# core/detector.py
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from typing import List, Tuple, Optional

class FaceDetector:
    def __init__(self, method: str, device: str = "cpu", config: dict = None):
        self.method = method
        self.device = device
        self.config = config or {}

        self.mtcnn = None
        self.haar = None

        if method == "haar":
            self.haar = cv2.CascadeClassifier(config.get("path", "haarcascade_frontalface_default.xml"))
            if self.haar.empty():
                raise ValueError(f"Failed to load Haar cascade: {config.get('path')}")
        elif method == "mtcnn":
            self.mtcnn = MTCNN(
                keep_all=True,
                device=device,
                selection_method=config.get("selection_method", "probability"),
                select_largest=False,
                min_face_size=config.get("min_face_size", 40),
                thresholds=tuple(config.get("thresholds", [0.6, 0.6, 0.7])),
                factor=config.get("factor", 0.709)
            )

    def detect(self, frame: np.ndarray) -> Tuple[List[List[int]], np.ndarray, Optional[List]]:
        """
        Returns: (boxes, probs, landmarks)
        - boxes: List of [x1, y1, x2, y2]
        - probs: Detection probabilities
        - landmarks: facial keypoints or None
        """
        if self.method == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(
                gray,
                scaleFactor=self.config.get("scale_factor", 1.1),
                minNeighbors=self.config.get("min_neighbors", 5),
                minSize=self.config.get("min_size", (40, 40))
            )
            boxes = [(x, y, x+w, y+h) for x, y, w, h in faces]
            probs = np.ones(len(boxes)) if boxes else np.array([])
            landmarks = [None] * len(boxes)
            return boxes, probs, landmarks

        else:  # mtcnn
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)

            if boxes is None:
                return [], np.array([]), []

            return [box.astype(int).tolist() for box in boxes], probs, landmarks

