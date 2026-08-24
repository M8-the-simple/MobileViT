"""
Build centroids script - extracts face embeddings and computes class centroids.

Usage:
    python -m data.build_centroids
"""

import numpy as np
import os
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from src import ComponentFactory, get_config


def load_image_paths(class_dir: str) -> list[str]:
    """
    Load all valid image paths from a class directory.

    Args:
        class_dir: Path to the folder containing images for one person.

    Returns:
        List of absolute paths to valid image files.
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_paths = []

    if not os.path.isdir(class_dir):
        return image_paths

    for f in os.listdir(class_dir):
        if f.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(class_dir, f))

    return sorted(image_paths)


def extract_embeddings_from_image(
    img_path: str,
    detector,
    preprocessor,
    model,
    prob_threshold: float = 0.5
) -> list[np.ndarray]:
    """
    Detect faces, preprocess, and extract embeddings from one image.

    Args:
        img_path: Path to the image file.
        detector: Face detector instance.
        preprocessor: Image preprocessor instance.
        model: Embedding model instance.
        prob_threshold: Minimum face detection probability.

    Returns:
        List of embedding arrays for all detected faces above threshold.
    """
    embeddings = []

    img = cv2.imread(img_path)
    if img is None:
        return embeddings

    if model.name.startswith("buffalo_sc"):
        embedding = model.embed(img)
        if embedding is not None:
            embeddings.append(embedding)
            return embeddings
        else:
            return embeddings  # No face detected

    # Detect faces
    boxes, probs, landmarks = detector.detect(img)

    if len(boxes) == 0:
        return embeddings

    # Extract embedding for each face above threshold
    for box, prob, land in zip(boxes, probs, landmarks):
        if prob < prob_threshold:
            continue

        preprocessed_img = preprocessor(img, landmarks=land)
        emb = model.embed(preprocessed_img)

        if emb is not None:
            embeddings.append(emb)

    return embeddings


def compute_centroid(embeddings: list[np.ndarray], min_embeddings: int = 10) -> Optional[np.ndarray]:
    """
    Compute robust centroid using median (less sensitive to outliers than mean).

    Args:
        embeddings: List of embedding arrays.
        min_embeddings: Minimum number of embeddings required.

    Returns:
        Centroid array, or None if not enough embeddings.
    """
    if len(embeddings) < min_embeddings:
        return None

    embs = np.array(embeddings)
    centroid = np.median(embs, axis=0)

    return centroid


def save_centroids(
    centroids: dict[str, np.ndarray],
    output_dir: str
) -> None:
    """
    Save all centroids atomically to disk.

    Args:
        centroids: Dictionary mapping class names to centroid arrays.
        output_dir: Directory to save centroid files.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not centroids:
        print("Warning: No centroids to save.")
        return

    # Stack all centroids and labels
    centroid_arrays = list(centroids.values())
    labels = list(centroids.keys())

    mean_embs = np.stack(centroid_arrays, axis=0)
    label_arrays = np.array(labels)

    # Save to disk
    centroid_path = os.path.join(output_dir, "centroid_znacajke.npy")
    label_path = os.path.join(output_dir, "oznake.npy")

    np.save(centroid_path, mean_embs)
    np.save(label_path, label_arrays)

    print(f"Saved {len(centroids)} centroids to {output_dir}")
    print(f"  Features: {centroid_path}")
    print(f"  Labels: {label_path}")


def process_class(
    class_name: str,
    class_path: str,
    detector,
    preprocessor,
    model,
    prob_threshold: float = 0.5,
    min_embeddings: int = 10
) -> tuple[Optional[np.ndarray], int]:
    """
    Process all images for a single class/person.

    Args:
        class_name: Name of the person (folder name).
        class_path: Path to the class directory.
        detector: Face detector instance.
        preprocessor: Image preprocessor instance.
        model: Embedding model instance.
        prob_threshold: Minimum face detection probability.
        min_embeddings: Minimum embeddings required for valid centroid.

    Returns:
        Tuple of (centroid array or None, number of embeddings extracted).
    """
    image_paths = load_image_paths(class_path)

    if not image_paths:
        print(f"  Warning: No valid images found in {class_path}")
        return None, 0

    print(f"  Found {len(image_paths)} images")

    all_embeddings = []

    for img_path in image_paths:
        embeddings = extract_embeddings_from_image(
            img_path,
            detector,
            preprocessor,
            model,
            prob_threshold
        )
        if len(embeddings) < 1:
            print(f"Nije pronađeno lice u slici: {img_path}")
        else:
            all_embeddings.extend(embeddings)

    print(f"  Extracted {len(all_embeddings)} embeddings")

    if len(all_embeddings) < min_embeddings:
        print(f"  Warning: Only {len(all_embeddings)} embeddings (need {min_embeddings})")
        return None, len(all_embeddings)

    centroid = compute_centroid(all_embeddings, min_embeddings)
    return centroid, len(all_embeddings)


def main(person: Optional[str] = None) -> None:
    """Main entry point for building centroids."""
    # Initialize components
    cfg = get_config()
    model = ComponentFactory.create_model()
    preprocessor = ComponentFactory.create_preprocessor()
    detector = ComponentFactory.create_detector()

    # Get paths from config with defaults
    dataset_path = cfg.dataset_path if hasattr(cfg, 'dataset_path') else "dataset"
    centroid_path = cfg.centroid_path if hasattr(cfg, 'centroid_path') else "centroids"
    dataset_dir = os.path.join(dataset_path, "train")
    centroids_dir = os.path.join(centroid_path, model.name)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {centroids_dir}")
    print(f"Model: {model.name}")
    print("-" * 50)

    # Get all class directories
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return

    if person is not None:
        print(f"Processing only person: {person}")
        if person not in os.listdir(dataset_dir):
            print(f"Error: Person '{person}' not found in dataset.")
            return
        class_names = [person]
    else:
        print("Processing all persons in dataset...")
        class_names = [
            name for name in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, name))
        ]

    if not class_names:
        print(f"Error: No class folders found in {dataset_dir}")
        return

    print(f"Found {len(class_names)} classes to process\n")
    centroids = {}
    if person is not None:
        prev_centroids = np.load(os.path.join(centroids_dir, "centroid_znacajke.npy"), allow_pickle=True)
        prev_labels = np.load(os.path.join(centroids_dir, "oznake.npy"), allow_pickle=True)
        for label, centroid in zip(prev_labels, prev_centroids):
            centroids[label] = centroid
        print("Loaded existing centroids for other classes.")
    stats = {"success": 0, "skipped": 0, "failed": 0}

    for class_name in tqdm(class_names, desc="Building centroids", unit="person"):
        class_path = os.path.join(dataset_dir, class_name)

        try:
            centroid, emb_count = process_class(
                class_name,
                class_path,
                detector,
                preprocessor,
                model,
                prob_threshold=0.5,
                min_embeddings=10
            )

            if centroid is not None:
                centroids[class_name] = centroid
                stats["success"] += 1
                print(f"  ✓ Centroid saved ({emb_count} embeddings)")
            else:
                stats["skipped"] += 1
                print(f"  ✗ Skipped (insufficient embeddings)")

        except Exception as e:
            stats["failed"] += 1
            print(f"  ✗ Error processing {class_name}: {e}")

    # Save all centroids at once
    print("-" * 50)
    print(f"Processing complete:")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")

    if centroids:
        save_centroids(centroids, centroids_dir)
    else:
        print("Warning: No centroids were generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Centroid generation script for face embeddings.')
    parser.add_argument('person', nargs='?', help='Optional: the option for creating a centroid for a specific person (folder name).')
    args = parser.parse_args()
    main(args.person)