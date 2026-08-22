"""Intra-class similarity test - measures how well same person is recognized."""
import os
from pathlib import Path
import cv2
import numpy as np
import time
from collections import Counter
from evaluation.tests import get_embedding_from_image
from src import get_config, ComponentFactory


# === SETUP ===
config = get_config()

# Load centroids
model_name = ComponentFactory.create_model().name
centroids_path = f"centroids/{model_name}"
embeddings = np.load(f"{centroids_path}/centroid_znacajke.npy")
labels = np.load(f"{centroids_path}/oznake.npy")
thresholds = config.models[config.backend].thresholds


def intra_class_test(person: str, max_images: int = 0,
                    mean_embeddings=None, mean_labels=None, silent: bool = False):
    """Test intra-class similarity (same person recognition).

    Args:
        person: Name of person to test
        max_images: Maximum number of images to test (0 = all)
        mean_embeddings: Pre-loaded embeddings (optional)
        mean_labels: Pre-loaded labels (optional)
        silent: If True, suppress detailed output

    Returns:
        dict with keys: similarities, accuracy, correct, tested, elapsed_times
    """
    threshold = thresholds.get(person, 0.5)

    if mean_embeddings is None or mean_labels is None:
        mean_embeddings = embeddings
        mean_labels = labels

    person_idx = np.where(mean_labels == person)[0]
    if len(person_idx) == 0:
        print(f"Person '{person}' not found in database!")
        return None

    centroid = mean_embeddings[person_idx][0]

    person_path = os.path.join("dataset/test", person)
    if not os.path.exists(person_path):
        print(f"Directory not found: {person_path}")
        return None

    images = [f for f in os.listdir(person_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    similarities = []
    correct = 0
    tested = 0
    elapsed_times = []

    if not silent:
        print(f"Testing {person} | Threshold: {threshold} | "
              f"Centroid norm: {np.linalg.norm(centroid):.4f}\n")

    if max_images == 0:
        max_images = len(images)

    for img_name in images[:max_images]:
        img_path = os.path.join(person_path, img_name)
        emb, elapsed = get_embedding_from_image(img_path)

        if emb is None:
            print(f"  ✗ No face detected: {img_name}")
            continue

        sim = float(np.dot(centroid, emb))
        similarities.append(sim)
        elapsed_times.append(elapsed)

        # Same person should have similarity > threshold
        predicted_label = person if sim > threshold else "Unknown"
        is_correct = (predicted_label == person)

        if is_correct:
            correct += 1
        tested += 1

        if not silent:
            print(f"  {img_name:35} → sim: {sim:.4f} → {predicted_label}")

    if tested == 0:
        print("No valid embeddings!")
        return None

    accuracy = correct / tested * 100

    if not silent:
        print("\n" + "=" * 60)
        print(f"RESULT FOR {person.upper()}")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{tested})")
        print(f"  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Max similarity: {max(similarities):.4f}")
        print(f"  Min similarity: {min(similarities):.4f}")
        print(f"  20th percentile: {np.percentile(similarities, 20):.4f}")
        if elapsed_times:
            print(f"  Avg embedding time: {np.mean(elapsed_times)*1000:.2f}ms")
        print("=" * 60)

    return {
        'similarities': similarities,
        'accuracy': accuracy,
        'correct': correct,
        'tested': tested,
        'elapsed_times': elapsed_times,
        'threshold': threshold
    }


def run_all_intra_class(max_images: int = 0, silent: bool = False):
    """Run intra-class test for all persons in test dataset."""
    results = {}
    similarities = []
    for person in os.listdir("dataset/test"):
        person_path = os.path.join("dataset/test", person)
        if not os.path.isdir(person_path):
            continue

        result = intra_class_test(person, max_images=max_images, silent=silent)
        if result:
            results[person] = result
            similarities.extend(result['similarities'])
    # Summary
    print("\n" + "=" * 60)
    print("INTRA-CLASS SUMMARY")
    print("=" * 60)

    total_correct = sum(r['correct'] for r in results.values())
    total_tested = sum(r['tested'] for r in results.values())
    overall_accuracy = total_correct / total_tested * 100 if total_tested > 0 else 0
    print(f"Mean similarity across all persons: {np.mean(similarities):.4f}")
    print(f"Overall accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_tested})")
    print("\nPer-person accuracy:")
    for person, result in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {person:20} {result['accuracy']:5.1f}% ({result['correct']}/{result['tested']})")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Intra-class similarity test')
    parser.add_argument('person', nargs='?', help='Person name to test')
    parser.add_argument('--max', '-m', type=int, default=0,
                       help='Maximum images to test (0=all)')
    parser.add_argument('--silent', '-s', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Test all persons')
    args = parser.parse_args()

    if args.all:
        run_all_intra_class(max_images=args.max, silent=args.silent)
    elif args.person:
        intra_class_test(args.person, max_images=args.max, silent=args.silent)
    else:
        print("Usage:")
        print("  python -m evaluation.tests.intra_class <person_name>")
        print("  python -m evaluation.tests.intra_class --all")
        print("  python -m evaluation.tests.intra_class --all --silent")
