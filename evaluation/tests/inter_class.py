"""Inter-class similarity test - measures how well different persons are distinguished."""
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
model = ComponentFactory.create_model()
detector = ComponentFactory.create_detector()
preprocessor = ComponentFactory.create_preprocessor()

# Load centroids
model_name = model.name
centroids_path = f"centroids/{model_name}"
embeddings = np.load(f"{centroids_path}/centroid_znacajke.npy")
labels = np.load(f"{centroids_path}/oznake.npy")
thresholds = config.models[config.backend].thresholds


def inter_class_test(test_person: str, reference_person: str, max_images: int = 0,
                    mean_embeddings=None, mean_labels=None, silent: bool = False):
    """Test inter-class similarity (different persons should not match).

    Args:
        test_person: Person whose images to test
        reference_person: Person whose centroid to compare against
        max_images: Maximum number of images to test (0 = all)
        mean_embeddings: Pre-loaded embeddings (optional)
        mean_labels: Pre-loaded labels (optional)
        silent: If True, suppress detailed output

    Returns:
        dict with keys: similarities, false_positives, tested, elapsed_times
    """
    if mean_embeddings is None or mean_labels is None:
        mean_embeddings = embeddings
        mean_labels = labels

    test_idx = np.where(mean_labels == test_person)[0]
    reference_idx = np.where(mean_labels == reference_person)[0]

    if len(test_idx) == 0:
        print(f"Test person '{test_person}' not found in database!")
        return None
    if len(reference_idx) == 0:
        print(f"Reference person '{reference_person}' not found in database!")
        return None

    # Use reference person's centroid
    centroid = mean_embeddings[reference_idx][0]

    test_dataset_path = os.path.join(config.dataset_path, "test")

    test_path = os.path.join(test_dataset_path, test_person)
    if not os.path.exists(test_path):
        print(f"Directory not found: {test_path}")
        return None

    images = [f for f in os.listdir(test_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if max_images == 0:
        max_images = len(images)

    # Use test_person's threshold
    threshold = thresholds.get(test_person, 0.5) if thresholds else 0.5

    similarities = []
    false_positives = 0
    tested = 0
    elapsed_times = []

    if not silent:
        print(f"Testing {test_person} vs centroid of {reference_person} | "
              f"Threshold: {threshold} | Centroid norm: {np.linalg.norm(centroid):.4f}\n")

    for img_name in images[:max_images]:
        img_path = os.path.join(test_path, img_name)
        emb, elapsed = get_embedding_from_image(img_path)

        if emb is None:
            print(f"  ✗ No face detected: {img_name}")
            continue

        sim = float(np.dot(centroid, emb))
        similarities.append(sim)
        elapsed_times.append(elapsed)

        # Different person should have similarity < threshold
        # False positive = incorrectly identifying as the reference person
        is_false_positive = (sim > threshold)
        if is_false_positive:
            false_positives += 1
        tested += 1

        if not silent:
            label = "Similar (FP)" if is_false_positive else "Not similar (TN)"
            print(f"  {img_name:35} → sim: {sim:.4f} → {label}")

    if tested == 0:
        print("No valid embeddings!")
        return None

    far = false_positives / tested * 100  # False Acceptance Rate

    if not silent:
        print("\n" + "=" * 60)
        print(f"RESULT: {test_person.upper()} vs {reference_person.upper()} CENTROID")
        print(f"  False positives: {false_positives}/{tested} ({far:.1f}%)")
        print(f"  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Max similarity: {max(similarities):.4f}")
        print(f"  Min similarity: {min(similarities):.4f}")
        print(f"  90th percentile: {np.percentile(similarities, 90):.4f}")
        if elapsed_times:
            print(f"  Avg embedding time: {np.mean(elapsed_times)*1000:.2f}ms")
        print("=" * 60)

    return {
        'similarities': similarities,
        'false_positives': false_positives,
        'tested': tested,
        'far': far,
        'elapsed_times': elapsed_times,
        'threshold': threshold
    }


def run_all_inter_class(test_person: str = None, max_images: int = 0, silent: bool = False):
    """Run inter-class test for all person pairs or one person against all others."""
    results = {}
    similarities = []
    test_dataset_path = os.path.join(config.dataset_path, "test")
    all_persons = [p for p in os.listdir(test_dataset_path)
                  if os.path.isdir(os.path.join(test_dataset_path, p))]

    if test_person:
        # Test one person against all others
        other_persons = [p for p in all_persons if p != test_person]
        for ref_person in other_persons:
            result = inter_class_test(test_person, ref_person,
                                     max_images=max_images, silent=silent)
            if result:
                results[f"{test_person}_vs_{ref_person}"] = result
    else:
        # Test all pairs
        for i, test_p in enumerate(all_persons):
            for ref_p in all_persons[i+1:]:
                result = inter_class_test(test_p, ref_p,
                                         max_images=max_images, silent=silent)
                if result:
                    results[f"{test_p}_vs_{ref_p}"] = result
                    similarities.extend(result['similarities'])

    # Summary
    print("\n" + "=" * 60)
    print("INTER-CLASS SUMMARY")
    print("=" * 60)

    total_fp = sum(r['false_positives'] for r in results.values())
    total_tested = sum(r['tested'] for r in results.values())
    overall_far = total_fp / total_tested * 100 if total_tested > 0 else 0
    print(f"Mean similarity across all pairs: {np.mean(similarities):.4f}")
    print(f"Overall FAR: {overall_far:.1f}% ({total_fp}/{total_tested})")
    print("\nPer-pair FAR:")
    for pair, result in sorted(results.items(), key=lambda x: -x[1]['far']):
        print(f"  {pair:40} {result['far']:5.1f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inter-class similarity test')
    parser.add_argument('test_person', nargs='?', help='Test person name')
    parser.add_argument('reference_person', nargs='?', help='Reference person (centroid) name')
    parser.add_argument('--max', '-m', type=int, default=0,
                       help='Maximum images to test (0=all)')
    parser.add_argument('--silent', '-s', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Test all person pairs')
    args = parser.parse_args()

    if args.all:
        run_all_inter_class(max_images=args.max, silent=args.silent)
    elif args.test_person and args.reference_person:
        inter_class_test(args.test_person, args.reference_person,
                        max_images=args.max, silent=args.silent)
    elif args.test_person:
        run_all_inter_class(test_person=args.test_person,
                          max_images=args.max, silent=args.silent)
    else:
        print("Usage:")
        print("  python -m evaluation.tests.inter_class <test_person> <reference_person>")
        print("  python -m evaluation.tests.inter_class <test_person> --all")
        print("  python -m evaluation.tests.inter_class --all")
