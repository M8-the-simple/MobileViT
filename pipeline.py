# pipeline/pipeline.py
import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass
import os
import time

from src import get_config, FaceDetector, EmbeddingModel, ImagePreprocessor, CentroidComparator, ComponentFactory,  RecognitionStats

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
        self.fps_text = 0.0
        self.detector = detector
        self.preprocessor = preprocessor
        self.comparator = comparator
        self.config = get_config()
        self.stats = RecognitionStats(self.config.temporal_window)
        self.frame_result = FrameResult(boxes=[], probs=np.array([]), landmarks=[], processed=None, embedding=None, label="Unknown", confidence=0.0)

    def process_frame(self, frame: np.ndarray, frame_idx: int, ground_truth: str = None) -> Optional[FrameResult]:
        """Process a single frame. Returns last result if no face detected."""
        # Only process every N frames
        if frame_idx % self.config.frame_skip != 0:
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
            self.stats.update(embedding, label, ground_truth)

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
                    if show:
                        print(f"Prosječni FPS: {self.fps_text:.2f}")
                    else:
                        print(f"Prosječni FPS: {frame_idx / (time.perf_counter() - start_video):.2f}")
                break

            frame_idx += 1
            fps = frame_idx / (time.perf_counter() - start_video)
            result = self.process_frame(frame, frame_idx, ground_truth)
            
            if len(self.stats.system_embedding) > 0:
                emb = self.stats.system_embedding
                label, confidence = self.comparator.compare(emb)

                self.stats._make_system_decision(label, ground_truth)

            if show and result is not None:
                self._draw_result(frame, result, frame_idx, fps)

            if show:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if show:
            cv2.destroyAllWindows()
        self.stats.print_report()

    def _draw_result(self, frame, result, frame_idx, fps):
        if result.boxes:
            x1, y1, x2, y2 = result.boxes[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{result.label} ({result.confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"{self.stats.get_system_recognized_person()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if frame_idx % 5 == 0:
            self.fps_text = fps
        cv2.putText(frame, f"FPS: {self.fps_text:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            


# === ENTRY POINT ===
def create_pipeline() -> FaceRecognitionPipeline:
    """Factory function to create configured pipeline."""
    cfg = get_config()

    model = ComponentFactory.create_model()
    detector = ComponentFactory.create_detector()
    preprocessor = ComponentFactory.create_preprocessor()

    # Load centroids
    centroids_path = f"{cfg.centroid_path}/{model.name}"
    embeddings = np.load(f"{centroids_path}/centroid_znacajke.npy")
    labels = np.load(f"{centroids_path}/oznake.npy")
    thresholds = cfg.models[cfg.backend].thresholds

    comparator = CentroidComparator(embeddings, labels, thresholds)

    return FaceRecognitionPipeline(model, detector, preprocessor, comparator)


if __name__ == "__main__":
    import argparse
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description='Run temporal analysis or real-time face recognition on camera feed'
    )
    parser.add_argument('source', nargs='?', help='Camera index (int) for camera source.')
    parser.add_argument('person', nargs='?', help='Ground truth for the person.')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Run temporal analysis on videos in test_videos/ folder')
    args = parser.parse_args()

    if not os.path.exists(cfg.centroid_path):
        print(f"Centroid path does not exist: {cfg.centroid_path}, please create centroids first using data/build_centroids.py")
        exit(1)

    if args.test:
        test_videos_dir = cfg.test_videos_path
        for person in os.listdir(test_videos_dir):
            if os.path.isdir(os.path.join(test_videos_dir, person)):
                person_path = os.path.join(test_videos_dir, person)
                for video_file in os.listdir(person_path):
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        video_path = os.path.join(person_path, video_file)
                        pipeline = create_pipeline()
                    pipeline.run_video(video_path, ground_truth=person, show=False)
    elif args.source is not None:
        pipeline = create_pipeline()
        pipeline.run_video(int(args.source), ground_truth=args.person, show=True)
        
    # else:
    #     print("Usage:")
    #     print("  python -m evaluation.tests.generate_labels <person_name>")
    #     print("  python -m evaluation.tests.generate_labels <person_name> --max 10")
    #     print("  python -m evaluation.tests.generate_labels --all")
                        