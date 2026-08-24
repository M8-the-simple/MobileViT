"""Generate y_true and y_score values for ROC curve analysis.

This script creates the ground truth labels and similarity scores needed
for ROC curve analysis. It computes:
- y_true: 1 for same person (positive), 0 for different person (negative)
- y_score: similarity score between test image and reference centroid
"""
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import time

from src import get_config, ComponentFactory
from evaluation.tests import get_embedding_from_image



# === SETUP ===
config = get_config()
model = ComponentFactory.create_model()
detector = ComponentFactory.create_detector()
preprocessor = ComponentFactory.create_preprocessor()

# Load centroids
model_name = model.name
centroids_path = f"{config.centroid_path}/{model_name}"
embeddings = np.load(f"{centroids_path}/centroid_znacajke.npy")
labels = np.load(f"{centroids_path}/oznake.npy")


def generate_labels_for_person(person: str, max_images: int = 0,
                              output_dir: str = None, silent: bool = False):
    """Generate y_true and y_score for a specific person.

    Args:
        person: Person name to generate labels for
        max_images: Maximum images per person to test (0 = all)
        output_dir: Output directory for .npy files (default: similarities/<person>)
        silent: Suppress progress output

    Returns:
        dict with y_true, y_score, and metadata
    """
    if output_dir is None:
        output_dir = f"evaluation/tests/similarities/{model.name}/{person}"

    os.makedirs(output_dir, exist_ok=True)

    # Get positive samples (intra-class - same person)
    if not silent:
        print(f"\n{'='*60}")
        print(f"Generating POSITIVES for {person}...")
        print(f"{'='*60}")

    from evaluation.tests.intra_class import intra_class_test
    result_pos = intra_class_test(person, max_images=max_images, silent=True)

    if result_pos is None or result_pos.get('similarities') is None:
        print(f"No positive samples found for {person}")
        return None

    positives = result_pos['similarities']
    avg_embedding_time = result_pos.get('elapsed_times', [])

    # Get negative samples (inter-class - different persons)
    if not silent:
        print(f"\n{'='*60}")
        print(f"Generating NEGATIVES for {person}...")
        print(f"{'='*60}")

    test_dataset_path = os.path.join(config.dataset_path, "test")

    all_persons = [p for p in os.listdir(test_dataset_path)
                   if os.path.isdir(os.path.join(test_dataset_path, p)) and p != person]

    negatives = []

    from evaluation.tests.inter_class import inter_class_test
    for other_person in all_persons:
        result_neg = inter_class_test(person, other_person,
                                     max_images=max_images, silent=True)
        if result_neg and result_neg.get('similarities'):
            negatives.extend(result_neg['similarities'])
            if result_neg.get('elapsed_times'):
                avg_embedding_time.extend(result_neg['elapsed_times'])

    if len(negatives) == 0:
        print(f"No negative samples found for {person}")
        return None

    # Create y_true and y_score arrays
    # y_true = 1 for same person (positive), 0 for different person (negative)
    y_true = np.array([1] * len(positives) + [0] * len(negatives))
    y_score = np.array(list(positives) + list(negatives))

    # Save to files
    np.save(os.path.join(output_dir, "y_score.npy"), y_score)
    np.save(os.path.join(output_dir, "y_true.npy"), y_true)

    if not silent:
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {person}")
        print(f"{'='*60}")
        print(f"  Positives: {len(positives)}")
        print(f"  Negatives: {len(negatives)}")
        print(f"  Total samples: {len(y_true)}")
        print(f"  Saved to: {output_dir}")

        # Stats
        print(f"\n  Positive scores:")
        print(f"    Mean: {np.mean(positives):.4f}")
        print(f"    Std:  {np.std(positives):.4f}")
        print(f"    Min:  {min(positives):.4f}")
        print(f"    Max:  {max(positives):.4f}")

        print(f"\n  Negative scores:")
        print(f"    Mean: {np.mean(negatives):.4f}")
        print(f"    Std:  {np.std(negatives):.4f}")
        print(f"    Min:  {min(negatives):.4f}")
        print(f"    Max:  {max(negatives):.4f}")

        if avg_embedding_time:
            print(f"\n  Avg embedding time: {np.mean(avg_embedding_time)*1000:.2f}ms")

    return {
        'y_true': y_true,
        'y_score': y_score,
        'positives': positives,
        'negatives': negatives,
        'output_dir': output_dir,
        'embedding_time_ms': np.mean(avg_embedding_time)*1000 if avg_embedding_time else None
    }


def generate_all_labels(max_images: int = 0, output_base: str = None):
    """Generate y_true and y_score for all persons in test dataset.

    Args:
        max_images: Maximum images per person to test
        output_base: Base directory for outputs

    Returns:
        dict mapping person name to result dict
    """
    results = {}

    test_dataset_path = os.path.join(config.dataset_path, "test")

    all_persons = [p for p in os.listdir(test_dataset_path)
                   if os.path.isdir(os.path.join(test_dataset_path, p))]

    print(f"\n{'='*60}")
    print(f"GENERATING LABELS FOR ALL PERSONS")
    print(f"  Persons: {len(all_persons)}")
    print(f"  Max images per person: {max_images if max_images > 0 else 'all'}")
    print(f"  Output base: {output_base}")
    print(f"{'='*60}\n")

    for i, person in enumerate(all_persons, 1):
        print(f"\n[{i}/{len(all_persons)}] Processing {person}...")
        if output_base:
            output_dir = os.path.join(output_base, person)
        else:
            output_dir = None
        result = generate_labels_for_person(person, max_images=max_images,
                                           output_dir=output_dir, silent=False)
        if result:
            results[person] = result

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")

    total_positives = sum(len(r['positives']) for r in results.values())
    total_negatives = sum(len(r['negatives']) for r in results.values())

    print(f"  Total persons processed: {len(results)}")
    print(f"  Total positive samples: {total_positives}")
    print(f"  Total negative samples: {total_negatives}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate y_true and y_score for ROC analysis'
    )
    parser.add_argument('person', nargs='?', help='Person name to generate labels for')
    parser.add_argument('--max', '-m', type=int, default=0,
                       help='Maximum images per person (0=all)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory (default: similarities/{model.name}/<person>)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Generate for all persons')
    parser.add_argument('--silent', '-s', action='store_true',
                       help='Suppress progress output')
    args = parser.parse_args()

    if args.all:
        generate_all_labels(max_images=args.max)
    elif args.person:
        generate_labels_for_person(args.person, max_images=args.max,
                                  output_dir=args.output, silent=args.silent)
    else:
        print("Usage:")
        print("  python -m evaluation.tests.generate_labels <person_name>")
        print("  python -m evaluation.tests.generate_labels <person_name> --max 10")
        print("  python -m evaluation.tests.generate_labels --all")
