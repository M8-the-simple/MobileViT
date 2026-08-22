"""ROC curve generation and analysis for face recognition evaluation.

This script generates ROC curves from pre-computed y_true and y_score values.
It calculates:
- ROC curve (TPR vs FPR)
- AUC (Area Under Curve)
- EER (Equal Error Rate)
- FAR/FRR at various thresholds
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src import ComponentFactory, get_config
cfg = get_config()
model = ComponentFactory.create_model()

def load_labels(person: str, input_dir: str = None):
    """Load y_true and y_score from numpy files.

    Args:
        person: Person name
        input_dir: Input directory (default: similarities/<person>)

    Returns:
        tuple: (y_true, y_score) or (None, None) if not found
    """
    if input_dir is None:
        input_dir = f"evaluation/tests/similarities/{person}"

    y_true_path = os.path.join(input_dir, "y_true.npy")
    y_score_path = os.path.join(input_dir, "y_score.npy")

    if not os.path.exists(y_true_path) or not os.path.exists(y_score_path):
        print(f"  ⚠️  Labels not found in {input_dir}")
        return None, None

    y_true = np.load(y_true_path)
    y_score = np.load(y_score_path)

    return y_true, y_score


def compute_roc_metrics(y_true, y_score):
    """Compute ROC curve and related metrics.

    Args:
        y_true: Ground truth labels (1 = same person, 0 = different person)
        y_score: Similarity scores

    Returns:
        dict with roc_curve data and metrics
    """
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Find EER (Equal Error Rate) - point where FAR = FRR
    # FRR = 1 - TPR, so we find where FPR = 1 - TPR
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_idx]
    eer_threshold = thresholds_roc[eer_idx]

    # Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    equal_precision_recall_idx = np.nanargmin(np.abs(precision - recall))
    epr = precision[equal_precision_recall_idx]
    epr_threshold = thresholds_pr[equal_precision_recall_idx] if equal_precision_recall_idx < len(thresholds_pr) else None

    # Find thresholds for specific FAR/TPR values
    def find_threshold_for_tpr(target_tpr):
        """Find threshold for given TPR."""
        idx = np.where(tpr >= target_tpr)
        if len(idx[0]) > 0:
            return thresholds_roc[idx[0][0]]
        return None

    def find_threshold_for_far(target_far):
        """Find threshold for given FAR."""
        idx = np.where(fpr <= target_far)
        if len(idx[0]) > 0:
            return thresholds_roc[idx[0][-1]]
        return None

    metrics = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds_roc': thresholds_roc,
        'roc_auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'epr': epr,
        'epr_threshold': epr_threshold,
        'precision': precision,
        'recall': recall,
        'thresholds_pr': thresholds_pr,
        'avg_precision': avg_precision,
        'threshold_90_tpr': find_threshold_for_tpr(0.90),
        'threshold_95_tpr': find_threshold_for_tpr(0.95),
        'threshold_01_far': find_threshold_for_far(0.01),
        'threshold_10_far': find_threshold_for_far(0.10),
    }

    return metrics


def plot_roc_curve(y_true, y_score, person: str = None, save_path: str = None,
                  show: bool = True):
    """Plot and optionally save ROC curve.

    Args:
        y_true: Ground truth labels
        y_score: Similarity scores
        person: Person name for title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    metrics = compute_roc_metrics(y_true, y_score)

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    # ROC Curve
    # ax1 = axes[0]
    # ax1.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2,
    #          label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    # ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    # #Mark EER point
    # ax1.scatter(metrics['eer'], 1 - metrics['eer'], color='red', s=100,
    #             zorder=5, label=f'EER = {metrics["eer"]:.3f}')

    # ax1.set_xlim([-0.05, 1.0])
    # ax1.set_ylim([0.0, 1.05])
    # ax1.set_xlabel('False Positive Rate (FAR)', fontsize=12)
    # ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    # title = f'{cfg.backend}'
    # if person:
    #     title += f' - {person} (AUC={metrics["roc_auc"]:.3f})'
    # ax1.set_title(title, fontsize=14)
    # ax1.legend(loc="lower right")
    # ax1.grid(True, alpha=0.3)

    # Precision-Recall Curve
    ax2 = axes
    ax2.plot(metrics['recall'], metrics['precision'], color='green', lw=2,
             label=f'PR curve (AP = {metrics["avg_precision"]:.3f})')
    ax2.set_xlim([-0.05, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    title = f'{cfg.backend}'
    if person:
        title += f' - {person} (AP={metrics["avg_precision"]:.3f})'
    ax2.set_title(title, fontsize=14)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved ROC curve to: {save_path}")


    if show:
        plt.show()

    return metrics


def print_metrics(metrics, person: str = None):
    """Print ROC metrics in a readable format."""
    print("\n" + "=" * 60)
    if person:
        print(f"ROC METRICS FOR: {person}")
    else:
        print("ROC METRICS")
    print("=" * 60)

    print(f"\n  AUC: {metrics['roc_auc']:.4f}")
    print(f"  EER: {metrics['eer']:.4f} (at threshold ≈ {metrics['eer_threshold']:.4f})")
    print(f"  Equal Precision-Recall: {metrics['epr']:.4f} (at threshold ≈ {metrics['epr_threshold']:.4f})")
    print(f"\n  Thresholds:")
    if metrics['threshold_90_tpr']:
        print(f"    For TPR ≥ 90%: threshold ≈ {metrics['threshold_90_tpr']:.4f}")
    if metrics['threshold_95_tpr']:
        print(f"    For TPR ≥ 95%: threshold ≈ {metrics['threshold_95_tpr']:.4f}")
    if metrics['threshold_01_far']:
        print(f"    For FAR ≤ 1%:  threshold ≈ {metrics['threshold_01_far']:.4f}")
    if metrics['threshold_10_far']:
        print(f"    For FAR ≤ 10%: threshold ≈ {metrics['threshold_10_far']:.4f}")

    print(f"\n  Average Precision: {metrics['avg_precision']:.4f}")
    print("=" * 60)


def roc_curve_for_person(person: str, input_dir: str = None, save_path: str = None,
                        show: bool = True, silent: bool = False):
    """Generate ROC curve for a specific person.

    Args:
        person: Person name
        input_dir: Input directory (default: similarities/<person>)
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        silent: Suppress detailed output

    Returns:
        dict with metrics or None if labels not found
    """
    y_true, y_score = load_labels(person, input_dir)

    if y_true is None or y_score is None:
        return None

    if not silent:
        print(f"\n  Positives: {np.sum(y_true == 1)}")
        print(f"  Negatives: {np.sum(y_true == 0)}")

    metrics = plot_roc_curve(y_true, y_score, person, save_path, show)

    if not silent:
        print_metrics(metrics, person)

    return metrics


def roc_curve_all(input_dir: str = f"evaluation/tests/similarities/{model.name}", save_dir: str = None,
                 show: bool = True, silent: bool = False):
    """Generate ROC curves for all persons with saved labels.

    Args:
        input_dir: Base directory containing person subdirectories
        save_dir: Directory to save ROC curves (optional)
        show: Whether to display plots
        silent: Suppress detailed output

    Returns:
        dict mapping person name to metrics
    """
    if not os.path.exists(input_dir):

        print(f"Input directory not found: {input_dir}")
        return {}

    results = {}

    persons = [d for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d))]

    if not silent:
        print(f"\n{'='*60}")
        print(f"GENERATING ROC CURVES FOR ALL PERSONS")
        print(f"  Persons found: {len(persons)}")
        print(f"{'='*60}\n")

    for person in persons:
        person_dir = os.path.join(input_dir, person)
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"roc_{person}.png")

        if not silent:
            print(f"\n[{len(results)+1}/{len(persons)}] {person}...")

        metrics = roc_curve_for_person(person, person_dir, save_path, show=show, silent=silent)
        if metrics:
            results[person] = metrics

    # Summary table
    if not silent:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Person':<20} {'AUC':>8} {'EER':>8} {'AP':>8} {'EPR':>8}")
        print("-" * 60)
        for person, m in sorted(results.items(), key=lambda x: -x[1]['roc_auc']):
            print(f"{person:<20} {m['roc_auc']:>8.4f} {m['eer']:>8.4f} {m['avg_precision']:>8.4f} {m['epr']:>8.4f}")
        print("-" * 60)

        overall_auc = np.mean([m['roc_auc'] for m in results.values()])
        overall_eer = np.mean([m['eer'] for m in results.values()])
        overall_epr = np.mean([m['epr'] for m in results.values()])
        print(f"{'AVERAGE':<20} {overall_auc:>8.4f} {overall_eer:>8.4f} {overall_epr:>8.4f}")
        print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate ROC curves for face recognition')
    parser.add_argument('person', nargs='?', help='Person name')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input directory (default: similarities/{model.name}/<person>)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file to save ROC curve')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Generate ROC curves for all persons')
    parser.add_argument('--save-dir', '-d', type=str, default=None,
                       help='Directory to save all ROC curves')
    parser.add_argument('--silent', '-s', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots', default=False)
    args = parser.parse_args()

    show = not args.no_show

    print("Variable show is set to:", show)

    if args.all:
        roc_curve_all(show=show, silent=args.silent, save_dir=args.save_dir)
    elif args.person:
        roc_curve_for_person(args.person, args.input, args.output, show=show, silent=args.silent)
    else:
        print("Usage:")
        print("  python -m evaluation.tests.classification_evaluation <person_name>")
        print("  python -m evaluation.tests.classification_evaluation <person_name> --output roc.png")
        print("  python -m evaluation.tests.classification_evaluation --all")
        print("  python -m evaluation.tests.classification_evaluation --all --save-dir roc_curves")
