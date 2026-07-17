# --- Detection comparison / tracking ---
import sys
import os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config

class DetectionTracker:
    """Prati i uspoređuje detekcije između različitih detektora."""

    def __init__(self):
        self.haar_count = 0
        self.mtcnn_count = 0
        self.frames_processed = 0
        self.haar_total_time = 0.0
        self.mtcnn_total_time = 0.0
        self.config = get_config()

    def track_haar(self, frame_bgr):
        """Detektira lice s Haar i vraća broj detekcija."""
        import cv2
        import time

        haar_cascade = cv2.CascadeClassifier(self.config.haar_config["path"])
        if haar_cascade.empty():
            raise IOError("Nije učitan Haar cascade!")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        start = time.perf_counter()
        faces = haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        elapsed = time.perf_counter() - start

        boxes = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
        self.haar_count += len(boxes)
        self.haar_total_time += elapsed
        return len(boxes), boxes

    def track_mtcnn(self, frame_bgr, device=None):
        """Detektira lice s MTCNN i vraća broj detekcija."""
        import time
        from facenet_pytorch import MTCNN
        import cv2

        device = "cuda" if torch.cuda.is_available() else "cpu" if device is None else device

        mtcnn = MTCNN(keep_all=True, device=device)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
        elapsed = time.perf_counter() - start

        if boxes is None:
            boxes = []
        self.mtcnn_count += len(boxes)
        self.mtcnn_total_time += elapsed
        return len(boxes), boxes

    def compare_both(self, frame_bgr, device=None):
        """Pokreće oba detektora i uspoređuje rezultate."""
        self.frames_processed += 1

        haar_faces, haar_boxes = self.track_haar(frame_bgr)
        mtcnn_faces, mtcnn_boxes = self.track_mtcnn(frame_bgr, device)

        return {
            "haar_count": haar_faces,
            "haar_boxes": haar_boxes,
            "mtcnn_count": mtcnn_faces,
            "mtcnn_boxes": mtcnn_boxes,
            "haar_time": self.haar_total_time / self.frames_processed,
            "mtcnn_time": self.mtcnn_total_time / self.frames_processed
        }

    def print_summary(self):
        """Ispisuje sažetak usporedbe."""
        print("\n" + "="*50)
        print("DETECTION COMPARISON SUMMARY")
        print("="*50)
        print(f"Frames processed: {self.frames_processed}")
        print(f"Haar faces found: {self.haar_count} (avg: {self.haar_count/max(1,self.frames_processed):.2f}/frame)")
        print(f"MTCNN faces found: {self.mtcnn_count} (avg: {self.mtcnn_count/max(1,self.frames_processed):.2f}/frame)")
        print(f"Haar avg time: {self.haar_total_time/max(1,self.frames_processed)*1000:.1f} ms")
        print(f"MTCNN avg time: {self.mtcnn_total_time/max(1,self.frames_processed)*1000:.1f} ms")
        print("="*50)


def compare_detectors_on_video(video_path, max_frames=None, device=None):
    """
    Uspoređuje Haar i MTCNN detekciju na videu.

    Args:
        video_path: Putanja do videa
        max_frames: Maksimalan broj frejmova za procesuiranje (None = cijeli video)
        device: Uređaj za MTCNN (np. 'cuda' ili 'cpu')

    Returns:
        DetectionTracker objekt sa statistikama
    """
    import cv2

    tracker = DetectionTracker()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Ne mogu otvoriti video: {video_path}")
        return None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracker.compare_both(frame, device)
        frame_idx += 1

        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: Haar={tracker.haar_count}, MTCNN={tracker.mtcnn_count}")

    cap.release()
    tracker.print_summary()
    return tracker

if __name__ == "__main__":
    # Primjer korištenja
    import os
    videos_dir = r"val_videos"
    for person in os.listdir(videos_dir):
        person_path = os.path.join(videos_dir, person)
        if not os.path.isdir(person_path):
            continue
        for video_path in os.listdir(person_path):
            print(f"\n[INFO] Usporedba detektora na videu: {video_path}")
            compare_detectors_on_video(os.path.join(person_path, video_path), max_frames=300, device='cpu')