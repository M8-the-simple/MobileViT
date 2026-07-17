# core/preprocessor.py
import cv2
import numpy as np
from typing import Optional, List

class ImagePreprocessor:
    def __init__(self, target_size: int = 112, use_clahe: bool = False,
                use_sharpen: bool = False, use_alignment: bool = True):
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.use_sharpen = use_sharpen
        self.use_alignment = use_alignment

        # ArcFace reference points
        self.ref_points = np.array([
            [30.2946, 51.6963],   # left eye
            [97.5318, 51.5014],   # right eye
            [64.0000, 82.7366],   # nose
            [48.5493, 107.5000],  # left mouth corner
            [82.0000, 107.5000]  # right mouth corner
        ], dtype=np.float32) * target_size / 112.0

    def __call__(self, img_bgr: np.ndarray, landmarks: Optional[List] = None) -> np.ndarray:
        """Process BGR image, return RGB image ready for embedding."""
        if img_bgr is None:
            return None

        img = img_bgr.copy()

        # Alignment
        if self.use_alignment and landmarks is not None:
            img = self._align_face(img, landmarks)

        # CLAHE
        if self.use_clahe:
            img = self._apply_clahe(img)

        # Sharpen
        if self.use_sharpen:
            img = cv2.filter2D(img, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))

        # Resize
        if img.shape[:2] != (self.target_size, self.target_size):
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _align_face(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face using landmarks."""
        try:
            src = np.array(landmarks, dtype=np.float32).reshape(5, 2)
            tform, _ = cv2.estimateAffinePartial2D(src, self.ref_points, method=cv2.LMEDS)

            if tform is None:
                return img

            # Anti-flip check
            if np.linalg.det(tform[:2, :2]) < 0:
                tform[0, 0] = -tform[0, 0]
                tform[0, 2] = self.target_size - tform[0, 2]

            return cv2.warpAffine(img, tform, (self.target_size, self.target_size),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return img

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
