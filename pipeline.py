# pipeline/pipeline.py
import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass
import os
import time

from config import get_config
from factory import ComponentFactory
from core.detector import FaceDetector
from core.model import EmbeddingModel
from core.preprocessor import ImagePreprocessor
from recognition.comparator import CentroidComparator
from recognition.stats import RecognitionStats

@dataclass
class FrameResult:
    boxes: list
    probs: np.ndarray
    landmarks: list
    processed: Optional[np.ndarray]
    embedding: Optional[np.ndarray]
    label: str
    confidence: float


class FaceRecognitionPipeline:
    """
    Main pipeline that ties everything together.
    Each component is injected via constructor (DI pattern).
    """

    def __init__(
        self,
        model: EmbeddingModel,
        detector: FaceDetector,
        preprocessor: ImagePreprocessor,
        comparator: CentroidComparator
    ):
        self.model = model
        self.detector = detector
        self.preprocessor = preprocessor
        self.comparator = comparator
        self.config = get_config()
        self.stats = RecognitionStats(self.config)
        self.frame_result = FrameResult(boxes=[], probs=np.array([]), landmarks=[], processed=None, embedding=None, label="Unknown", confidence=0.0)

    def process_frame(self, frame: np.ndarray, frame_idx: int, ground_truth: str = None) -> Optional[FrameResult]:
        """Process a single frame. Returns last result if no face detected."""
        # Only process every N frames
        if frame_idx % self.config.frame_skip != 0 and frame_idx != 1:
            return self.frame_result  # Return last result if skipping

        self.frame_result.boxes = []
        # Detection
        detection_start_time = time.perf_counter()
        boxes, probs, landmarks = self.detector.detect(frame)
        self.stats.add_detection_time(time.perf_counter() - detection_start_time)

        if len(boxes) == 0:
            return self.frame_result  # No face detected, return last result

        # Take highest confidence face
        best_idx = np.argmax(probs) if len(probs) > 0 else 0
        box = boxes[best_idx]
        land = landmarks[best_idx] if len(landmarks) > 0 else None

        
        processed = self.preprocessor(frame, land)

        if processed is None:
            return self.frame_result  # Preprocessing failed, return last result

        # Embed
        embedding_start_time = time.perf_counter()
        embedding = self.model.embed(processed)
        self.stats.add_embedding_time(time.perf_counter() - embedding_start_time)

        # Recognize
        label, confidence = self.comparator.compare(embedding)
        if label is not None:
            self.stats.update(label, ground_truth)

        self.frame_result = FrameResult(
            boxes=[box],
            probs=probs,
            landmarks=landmarks,
            processed=processed,
            embedding=embedding,
            label=label,
            confidence=confidence,
        )

        return self.frame_result

    def run_video(self, source, ground_truth: Optional[str] = "Unknown", show=True):
        """Run pipeline on video file or camera."""
        if isinstance(source, str):
            cap = cv2.VideoCapture(source)
        elif isinstance(source, int):
            cap = cv2.VideoCapture(source)
            ground_truth = "Mathias"
        else:
            raise ValueError("source must be video path (str) or camera index (int)")

        frame_idx = 0
        start_video = time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_idx != 0:
                    print(f"Prosječni FPS: {frame_idx / (time.perf_counter() - start_video):.2f}")
                break

            frame_idx += 1
            result = self.process_frame(frame, frame_idx, ground_truth)
            
            if show and result is not None:
                self._draw_result(frame, result)

            if show:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if show:
            cv2.destroyAllWindows()
        self.stats.print_report()

    def _draw_result(self, frame, result):
        if result.boxes:
            x1, y1, x2, y2 = result.boxes[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{result.label} ({result.confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"{self.stats.get_system_recognized_person()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# === ENTRY POINT ===
def create_pipeline() -> FaceRecognitionPipeline:
    """Factory function to create configured pipeline."""
    cfg = get_config()

    model = ComponentFactory.create_model()
    detector = ComponentFactory.create_detector()
    preprocessor = ComponentFactory.create_preprocessor()

    # Load centroids
    centroids_path = f"centroids/{cfg.models[cfg.backend].name.replace(':', '_').replace('/', '_')}"
    embeddings = np.load(f"{centroids_path}/centroid_znacajke.npy")
    labels = np.load(f"{centroids_path}/oznake.npy")
    thresholds = cfg.models[cfg.backend].thresholds

    comparator = CentroidComparator(embeddings, labels, thresholds)

    return FaceRecognitionPipeline(model, detector, preprocessor, comparator)


if __name__ == "__main__":
    # videos_path = r"C:\Users\matia\Documents\RiTeh\6_semestar\Zavrsni_rad\HaarCascade\MobileViT\val_videos"
    # for person_folder in os.listdir(videos_path):
    #     person_path = os.path.join(videos_path, person_folder)
    #     if os.path.isdir(person_path):
    #         for video_file in os.listdir(person_path):
    #             if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
    #                 video_path = os.path.join(person_path, video_file)
    #                 print(f"Processing video: {video_path}")
    pipeline = create_pipeline()
    pipeline.run_video(0)  # Camera, or: pipeline.run_video("video.mp4")
    