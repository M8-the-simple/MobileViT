import cv2
import os
from pathlib import Path
from detectors import FaceDetector

detector = FaceDetector(method="mtcnn")

def extract_faces_from_video(video_path: str, output_dir: str, interval=8, min_conf=0.98):
    """
    Ekstrahira lica iz videa samo ako je MTCNN confidence >= min_conf (preporučeno 0.98)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ne mogu otvoriti video: {video_path}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Kraj videa.")
            break

        if count % interval == 0:
            # return_prob=True da dobijemo i vjerojatnost
            boxes, probs = detector.detect(frame)
        
            if boxes is not None and probs is not None:
                # Provjeri ima li barem jedno lice iznad praga
                high_conf_probs = [p for p in probs if p is not None and p >= min_conf]
                
                if high_conf_probs:
                    best_prob = max(high_conf_probs)
                    
                    filename = f"frame_{count:06d}_conf{best_prob:.4f}.jpg"
                    cv2.imwrite(str(output_path / filename), frame)  # ← cijeli frame
                    
                    saved_count += 1
                    
                    if saved_count % 15 == 0 or best_prob < 0.99:
                        print(f"Spremljen frame {saved_count:4d} | frame {count:6d} | "
                              f"conf {best_prob:.4f} | veličina {frame.shape[1]}x{frame.shape[0]}")
        count += 1

    cap.release()
    print(f"\nZavršeno! Spremljeno ukupno {saved_count} lica (min conf {min_conf}).")


if __name__ == "__main__":
    timestamp = "165564"
    video_path = rf"C:\Users\matia\Documents\RiTeh\6_semestar\Zavrsni_rad\Mathias\video_Mathias_20260518_{timestamp}.mp4"
    
    extract_faces_from_video(
        video_path=video_path,
        output_dir=r"dataset\train\Mathias",
        interval=6,       # svaki 6. frame (možeš povećati ako je video dug)
        min_conf=0.98
    )