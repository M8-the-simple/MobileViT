import cv2
import os
from pathlib import Path
from src import get_config, ComponentFactory

cfg = get_config()
detector = ComponentFactory.create_detector()

def extract_faces_from_video(video_path: str, output_dir: str, interval, min_conf): # defaults: interval=8, min_conf=0.98
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
            boxes, probs, _ = detector.detect(frame)

            if boxes is not None and probs is not None:
                # Provjeri ima li barem jedno lice iznad praga
                high_conf_probs = [p for p in probs if p is not None and p >= min_conf]

                if high_conf_probs:
                    best_prob = max(high_conf_probs)

                    filename = f"frame_{count:06d}_conf{best_prob:.4f}.jpg"
                    #cv2.imwrite(str(output_path / filename), frame)  # ← cijeli frame

                    saved_count += 1

                    if saved_count % 15 == 0 or best_prob < 0.99:
                        print(f"Spremljen frame {saved_count:4d} | frame {count:6d} | "
                              f"conf {best_prob:.4f} | veličina {frame.shape[1]}x{frame.shape[0]}")
        count += 1

    cap.release()
    print(f"\nZavršeno! Spremljeno ukupno {saved_count} lica (min conf {min_conf}).")


if __name__ == "__main__":
    import argparse
    from src import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description='Extract frames from train or test videos'
    )
    parser.add_argument('mode', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('person', nargs='?', help='Specific person to process')
    parser.add_argument('--remove', action='store_true', help='Remove old frames before extraction')

    args = parser.parse_args()

    # Resolve paths based on mode and arguments
    if args.mode == 'train':
        video_dir = cfg.train_videos_path
        output_root = f"{cfg.dataset_path}/train"
    else:
        video_dir = cfg.test_videos_path
        output_root = f"{cfg.dataset_path}/test"

    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        exit(1)
        
    for class_name in os.listdir(video_dir):
        if args.person:
            if class_name != args.person:
                continue  # Skip other persons if a specific one is provided
        class_path = os.path.join(video_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        print(f"Započinjem ekstrackiju frame-ova za: {class_name}")
        video_paths = [os.path.join(class_path, f) for f in os.listdir(class_path)
                    if f.lower().endswith((".mp4", ".avi"))]

        if len(video_paths) == 0:
            print(f"Nema video datoteka u folderu: {class_path}")
            continue

        output_dir = os.path.join(output_root, class_name)
        output_path = Path(output_dir)

        if args.remove and output_path.exists():
            if any(output_path.iterdir()):
                print(f"Brišem stare frame-ove iz {output_dir}...")
                for file in output_path.iterdir():
                    os.remove(file)

        for video_path in video_paths:
            extract_faces_from_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    interval=4,
                    min_conf=0.95
                )
