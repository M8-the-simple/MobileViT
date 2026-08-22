"""
PCA Visualization Script for Face Embeddings

This script performs PCA (Principal Component Analysis) on face embeddings
to visualize them in 2D. It can work with either:
1. Individual face embeddings from the test dataset
2. Pre-computed class centroids

Usage:
    python -m src.analysis.make_pca --help                    # Show help
    python -m src.analysis.make_pca --mode faces             # PCA on test dataset faces
    python -m src.analysis.make_pca --mode centroids          # PCA on class centroids
    python -m src.analysis.make_pca --mode all --save        # Run both and save plots
    python -m src.analysis.make_pca --mode faces --verbose   # More detailed output
"""

import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np
import cv2
from pathlib import Path

from src import ComponentFactory, get_config

cfg = get_config()
detector = ComponentFactory.create_detector()
preprocessor = ComponentFactory.create_preprocessor()
model = ComponentFactory.create_model()

COLORS = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']


def get_embedding_from_image(image_path: str, verbose: bool = False) -> np.ndarray | None:
    """
    Load image, detect face, preprocess, and extract embedding.

    Args:
        image_path: Path to the image file
        verbose: Whether to print detailed progress

    Returns:
        embedding array, or None if face detection fails
    """
    img = cv2.imread(image_path)
    if img is None:
        if verbose:
            print(f"  ⚠️  Cannot read image: {image_path}")
        return None

    if model.name.startswith("buffalo_sc"):
        embedding = model.embed(img)
        if embedding is not None:
            return embedding
        else:
            if verbose:
                print(f"  ⚠️  No face detected in image: {image_path}")
            return None

    boxes, probs, landmarks = detector.detect(img)

    if len(boxes) == 0:
        if verbose:
            print(f"  ⚠️  No face detected: {image_path}")
        return None

    # Take highest confidence face
    best_idx = np.argmax(probs) if len(probs) > 0 else 0
    box = boxes[best_idx]
    land = landmarks[best_idx] if landmarks is not None and len(landmarks) > best_idx else None

    # Preprocess
    processed = preprocessor(img, land)

    if processed is None:
        if verbose:
            print(f"  ⚠️  Preprocessing failed: {image_path}")
        return None

    # Extract embedding
    return model.embed(processed)


def run_pca(embeddings: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """
    Run PCA on embeddings.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        n_components: Number of PCA components (default: 2)

    Returns:
        Tuple of (transformed_2d_embeddings, fitted_pca_object)
    """
    n_samples, n_features = embeddings.shape

    # Adjust n_components if needed
    actual_components = min(n_components, n_samples - 1, n_features)
    if actual_components < n_components:
        print(f"  ℹ️  Adjusting n_components from {n_components} to {actual_components} "
              f"(samples={n_samples}, features={n_features})")

    pca = PCA(n_components=actual_components)
    embeddings_transformed = pca.fit_transform(embeddings)

    return embeddings_transformed, pca

def make_pca_plot_1d(embeddings_1d: np.ndarray,
                     labels: np.ndarray,
                     title: str,
                     xlabel: str = "PC1",
                     explained_variance: np.ndarray | None = None) -> plt.Figure:
    """
    Plot class centroids in 1-D PCA space.

    Args:
        embeddings_1d: Array of shape (n_classes,) or (n_classes, 1)
                       – already the centroids projected to 1-D
        labels: Array of class names corresponding to each centroid
        title: Plot title
        xlabel: X-axis label
        explained_variance: Explained variance ratio (optional)

    Returns:
        matplotlib Figure
    """
    embeddings_1d = np.asarray(embeddings_1d).ravel()

    fig, ax = plt.subplots(figsize=(10, 3))

    for i, (x, class_name) in enumerate(zip(embeddings_1d, labels)):
        color = COLORS[i % len(COLORS)]

        # Plot the centroid
        ax.scatter(x, 0,
                   s=180,
                   color=color,
                   marker='X',
                   edgecolors='k',
                   linewidth=1.5,
                   zorder=5,
                   label=class_name)

        # Label next to the marker
        ax.annotate(class_name,
                    (x, 0.08),
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold')

    # Axis formatting
    if explained_variance is not None:
        xlabel = f"{xlabel} ({explained_variance[0]*100:.1f}% var)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_yticks([])
    ax.set_ylim(-0.3, 0.4)
    #ax.set_title(title, fontsize=14)
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    return fig


def make_pca_plot(embeddings_2d: np.ndarray,
                  labels: np.ndarray,
                  title: str,
                  xlabel: str,
                  ylabel: str,
                  explained_variance: np.ndarray | None = None,
                  annotate_points: bool = True) -> plt.Figure:
    """
    Create a scatter plot of embeddings in 2D.

    Args:
        embeddings_2d: Array of shape (n_samples, 2)
        labels: Array of class labels for each embedding
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        explained_variance: Explained variance ratio for each PC
        annotate_points: Whether to annotate each point with its label

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    unique_labels = np.unique(labels)

    for i, class_name in enumerate(unique_labels):
        mask = labels == class_name
        class_2d = embeddings_2d[mask]

        color = COLORS[i % len(COLORS)]

        # Scatter plot for this class
        ax.scatter(class_2d[:, 0],
                   class_2d[:, 1],
                   c=color,
                   
                   label=class_name,
                   edgecolors="none",
                   s=22,
                   alpha=0.9)

        # Annotate each point
        # if annotate_points:
        #     for j, (x, y) in enumerate(class_2d):
        #         offset = 0.02 * (j % 3 - 1)  # Slight offset to avoid overlapping labels
        #         ax.annotate(f"{class_name}_{j}",
        #                    (x + offset, y + offset),
        #                    fontsize=7,
        #                    alpha=0.7)

        # Mark centroid with X
        if len(class_2d) > 1:
            centroid = class_2d.mean(axis=0)
            ax.scatter(centroid[0],
                       centroid[1],
                       c=color,
                       edgecolors='k',
                       marker='X',
                       s=300,
                       linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    return fig


def create_pca_for_centroids(save: bool = False, verbose: bool = False) -> plt.Figure | None:
    """
    Create PCA visualization for pre-computed class centroids.

    Expected files in centroids/{model_name}/:
        - centroid_znacajke.npy: Array of centroid embeddings
        - oznake.npy: Array of class labels

    Args:
        save: Whether to save the plot to a file
        verbose: Whether to print detailed progress

    Returns:
        matplotlib Figure object, or None if centroids not found
    """
    centroids_path = Path("centroids") / model.name

    if not centroids_path.exists():
        print(f"❌ Centroids directory not found: {centroids_path}")
        return None

    centroid_file = centroids_path / "centroid_znacajke.npy"
    labels_file = centroids_path / "oznake.npy"

    if not centroid_file.exists() or not labels_file.exists():
        print(f"❌ Required files not found in {centroids_path}")
        print(f"   Expected: {centroid_file.name}, {labels_file.name}")
        return None

    embeddings = np.load(centroid_file)
    labels = np.load(labels_file, allow_pickle=True)

    n_samples, n_features = embeddings.shape
    print(f"✅ Loaded {n_samples} centroids, {n_features} dimensions")

    if n_samples < 2:
        print("❌ Not enough centroids for PCA (need at least 2)")
        return None

    # Run PCA
    embeddings_2d, pca = run_pca(embeddings, n_components=2)

    explained = pca.explained_variance_ratio_
    total_variance = sum(explained)
    
    print(f"📊 Explained variance: PC1={explained[0]:.2%}, #PC2={explained[1]:.2%}, Total={total_variance:.2%}")

    # Create plot
    title = f'PCA of Face Centroids ({cfg.backend})'
    xlabel = f'Principal Component 1 ({explained[0]:.1%})'
    ylabel = f'Principal Component 2 ({explained[1]:.1%})'
    
    fig = make_pca_plot(embeddings_2d, labels, title, xlabel, ylabel, explained)

    if save:
        output_path = centroids_path / "centroids_pca.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved plot to: {output_path}")

    return fig


def create_pca_for_faces(dataset_path: str = "dataset/test",
                         save: bool = False,
                         verbose: bool = False) -> plt.Figure | None:
    """
    Create PCA visualization for all faces in the test dataset.

    Args:
        dataset_path: Path to the test dataset directory
        save: Whether to save the plot to a file
        verbose: Whether to print detailed progress

    Returns:
        matplotlib Figure object, or None if dataset not found
    """
    person_path = Path(dataset_path)

    if not person_path.exists():
        print(f"❌ Dataset directory not found: {person_path}")
        return None

    all_embeddings = []
    all_labels = []

    class_dirs = [d for d in person_path.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"❌ No class directories found in: {person_path}")
        return None

    print(f"📁 Processing {len(class_dirs)} classes...")

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir()
                 if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

        if verbose:
            print(f"\n  Processing {class_name}: {len(images)} images")

        class_embeddings = []

        for img_path in images:
            emb = get_embedding_from_image(str(img_path), verbose=verbose)
            if emb is not None:
                class_embeddings.append(emb)
                all_embeddings.append(emb)
                all_labels.append(class_name)

        print(f"  ✅ {class_name}: {len(class_embeddings)}/{len(images)} embeddings extracted")

    if len(all_embeddings) < 2:
        print("❌ Not enough embeddings for PCA (need at least 2)")
        return None

    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels)

    print(f"📊 Total: {len(all_embeddings)} embeddings, {embeddings.shape[1]} dimensions")

    # Run PCA
    embeddings_2d, pca = run_pca(embeddings, n_components=2)

    explained = pca.explained_variance_ratio_
    total_variance = sum(explained)
    print(f"📊 Explained variance: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}, Total={total_variance:.2%}")

    # Create plot
    title = f''
    xlabel = f'Principal Component 1 ({explained[0]:.1%})'
    ylabel = f'Principal Component 2 ({explained[1]:.1%})'

    fig = make_pca_plot(embeddings_2d, labels, title, xlabel, ylabel, explained)

    if save:
        output_path = Path("outputs") / f"faces_pca_{cfg.backend}.png"
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved plot to: {output_path}")

    return fig


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='PCA Visualization for Face Embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.analysis.pca --mode faces
  python -m src.analysis.pca --mode centroids
  python -m src.analysis.pca --mode all --save
  python -m src.analysis.pca --mode faces --dataset dataset/train --verbose
        """
    )

    parser.add_argument(
        '--mode',
        choices=['faces', 'centroids', 'all'],
        default='all',
        help='What to visualize: faces from dataset, pre-computed centroids, or both (default: all)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/test',
        help='Path to dataset directory (default: dataset/test)'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save plots to files instead of displaying'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not show plots (useful with --save)'
    )

    args = parser.parse_args()

    print(f"🤖 Using model: {model.name}")
    print(f"📋 Mode: {args.mode}")
    print("-" * 50)

    if args.mode in ('faces', 'all'):
        print("\n=== PCA on Face Embeddings ===")
        fig = create_pca_for_faces(
            dataset_path=args.dataset,
            save=args.save,
            verbose=args.verbose
        )
        if fig and not args.no_display:
            plt.show()

    if args.mode in ('centroids', 'all'):
        print("\n=== PCA on Centroids ===")
        fig = create_pca_for_centroids(
            save=args.save,
            verbose=args.verbose
        )
        if fig and not args.no_display:
            plt.show()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
