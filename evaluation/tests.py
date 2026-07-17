# tests.py - Refactored to use new modular structure
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from factory import ComponentFactory
from core.detector import FaceDetector
from core.model import EmbeddingModel
from core.preprocessor import ImagePreprocessor
from recognition.comparator import CentroidComparator


# === SETUP ===
config = get_config()
model = ComponentFactory.create_model()
detector = ComponentFactory.create_detector()
preprocessor = ComponentFactory.create_preprocessor()

# Load centroids
model_name = config.models[config.backend].name.replace(':', '_').replace('/', '_')
centroids_path = f"centroids/{model_name}"
embeddings = np.load(f"{centroids_path}/centroid_znacajke.npy")
labels = np.load(f"{centroids_path}/oznake.npy")
thresholds = config.models[config.backend].thresholds

def get_embedding_from_image(image_path: str) -> np.ndarray:
    """
    Load image, detect face, preprocess, and extract embedding.
    Returns None if no face detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️  Ne mogu otvoriti sliku: {image_path}")
        return None, None

    boxes, probs, landmarks = detector.detect(img)

    if len(boxes) == 0:
        print(f"⚠️  Nije detektirano lice: {image_path}")
        return None, None

    # Take highest confidence face
    best_idx = np.argmax(probs) if len(probs) > 0 else 0
    box = boxes[best_idx]
    land = landmarks[best_idx] if landmarks is not None and len(landmarks) > best_idx else None

    # Preprocess
    processed = preprocessor(img, land)

    if processed is None:
        return None, None

    # Extract embedding
    before_emb = time.perf_counter()
    emb = model.embed(processed)
    elapsed_emb = time.perf_counter() - before_emb
    return emb, elapsed_emb


def intra_class_test(person: str, max_images: int = 0,
                    mean_embeddings=None, mean_labels=None, silent: bool = False):
    """Test intra-class similarity (same person)."""
    threshold = thresholds.get(person, 0.5)

    if mean_embeddings is None or mean_labels is None:
        mean_embeddings = embeddings
        mean_labels = labels

    person_idx = np.where(mean_labels == person)[0]
    if len(person_idx) == 0:
        print("Osoba nije pronađena u bazi!")
        return

    centroid = mean_embeddings[person_idx][0]

    person_path = os.path.join("dataset/val", person)
    if not os.path.exists(person_path):
        print(f"Nema direktorija: {person_path}")
        return

    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    similarities = []
    correct = 0
    tested = 0
    avg_embedding_time = []

    print(f"Testiram {person} | Threshold: {threshold} | Centroid norm: {np.linalg.norm(centroid):.4f}\n")

    if max_images == 0:
        max_images = len(images)

    for img_name in images[:max_images]:
        img_path = os.path.join(person_path, img_name)
        emb, elapsed = get_embedding_from_image(img_path)
        avg_embedding_time.append(elapsed)

        if emb is None:
            print(f"✗ Nije detektirano lice: {img_name}")
            continue

        sim = float(np.dot(centroid, emb))
        similarities.append(sim)

        label = person if sim > threshold else "Unknown"
        is_correct = (label == person)

        if is_correct:
            correct += 1
        tested += 1

        if not silent:
            print(f"{img_name:35} → sim: {sim:.4f} → {label}")

    if tested == 0:
        print("Nema validnih embeddinga!")
        return

    accuracy = correct / tested * 100
    if not silent:
        print("\n" + "=" * 60)
        print(f"REZULTAT ZA {person.upper()}")
        print(f"Točnost: {accuracy:.1f}% ({correct}/{tested})")
        print(f"Prosječna sličnost: {np.mean(similarities):.4f}")
        print(f"Max sličnost: {max(similarities):.4f}")
        print(f"Min sličnost: {min(similarities):.4f}")
        print(f"20. percentil: {np.percentile(similarities, 20):.4f}")
        print("=" * 60)

    return similarities, elapsed


def inter_class_test(person1: str, person2: str, max_images: int = 0,
                    mean_embeddings=None, mean_labels=None, silent: bool = False):
    """Test inter-class similarity (different persons)."""

    if mean_embeddings is None or mean_labels is None:
        mean_embeddings = embeddings
        mean_labels = labels

    person1_idx = np.where(mean_labels == person1)[0]
    person2_idx = np.where(mean_labels == person2)[0]

    if len(person1_idx) == 0 or len(person2_idx) == 0:
        print("Osoba nije pronađena u bazi!")
        return

    centroid = mean_embeddings[person2_idx][0]

    person_path = os.path.join("dataset/val", person1)
    if not os.path.exists(person_path):
        print(f"Nema direktorija: {person_path}")
        return

    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if max_images == 0:
        max_images = len(images)

    similarities = []
    similar = 0
    tested = 0

    threshold = thresholds.get(person1, 0.5)
    print(f"Testiram {person1} | Threshold: {threshold} | Centroid norm: {np.linalg.norm(centroid):.4f}\n s Osobom: {person2}")

    for img_name in images[:max_images]:
        img_path = os.path.join(person_path, img_name)
        emb, elapsed = get_embedding_from_image(img_path)

        if emb is None:
            print(f"✗ Nije detektirano lice: {img_name}")
            continue

        sim = float(np.dot(centroid, emb))
        similarities.append(sim)

        label = "Not similar" if sim < threshold else "Similar"
        is_similar = (label == "Similar")

        if is_similar:
            similar += 1
        tested += 1

        if not silent:
            print(f"{img_name:35} → sim: {sim:.4f} → {label}")

    if tested == 0:
        print("Nema validnih embeddinga!")
        return

    if not silent:
        accuracy = similar / tested * 100
        print("\n" + "=" * 60)
        print(f"REZULTAT ZA {person1.upper()} U USPOREDBI S {person2.upper()}")
        print(f"Sličnost: {accuracy:.1f}% ({similar}/{tested})")
        print(f"Prosječna sličnost: {np.mean(similarities):.4f}")
        print(f"Max sličnost: {max(similarities):.4f}")
        print(f"Min sličnost: {min(similarities):.4f}")
        print(f"90. percentil: {np.percentile(similarities, 90):.4f}")
        print("=" * 60)

    return similarities, elapsed


def similarity_to_centroid(person, centroid_embeddings=None, labels_arr=None):
    """Test similarity of all images to all centroids."""

    recognized_people = Counter()

    if centroid_embeddings is None or labels_arr is None:
        centroid_embeddings = embeddings
        labels_arr = labels

    person_idx = np.where(labels_arr == person)[0]
    if len(person_idx) == 0:
        print("Osoba nije pronađena u bazi!")
        return

    person_path = os.path.join("dataset/val", person)
    if not os.path.exists(person_path):
        print(f"Nema direktorija: {person_path}")
        return

    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    similarities = []
    self_similarities = []
    print(len(images), "slika za osobu:", person)

    for img_name in images:
        img_path = os.path.join(person_path, img_name)
        emb, _ = get_embedding_from_image(img_path)

        if emb is None:
            print(f"✗ Nije detektirano lice: {img_name}")
            continue

        sim = np.dot(centroid_embeddings, emb)
        idx = np.argmax(sim)
        similarities.append(sim[idx])
        self_similarities.append(sim[person_idx][0])
        recognized_people[labels_arr[idx]] += 1
        print(f"{img_name:35} → Maksimalna sličnost prema centroidu: {sim[idx]:.4f} -> {labels_arr[idx]}")

    if len(similarities) != 0:
        print("=" * 60)
        print("Prosječna maksimalna sličnost prema centroidu:", np.mean(similarities))
        print("Prosječna sličnost prema vlastitom centroidu:", np.mean(self_similarities))
        print("Prepoznate osobe prema centroidu:", recognized_people.most_common(3))
        print("=" * 60)

    return similarities


def make_labels_and_scores(original_person):
    from sklearn.metrics import roc_curve, auc

    # Positives = intra-class similarity (same person)
    # Negatives = inter-class similarity (different persons)

    avg_embedding_time = []

    positives, emb_time = intra_class_test(original_person, silent=True)

    avg_embedding_time.append(emb_time)

    if positives is None:
        print(f"Nema pozitivnih primjera za: {original_person}")
        return

    negatives = []

    for inter_person in os.listdir(r"dataset\val"):
        if inter_person != original_person:
            inter_sims, emb_time = inter_class_test(original_person, inter_person, silent=True)
            if inter_sims is not None:
                negatives.extend(inter_sims)
                avg_embedding_time.append(emb_time)

    if len(negatives) == 0:
        print(f"Nema negativnih primjera za: {original_person}")
        return

    # Prepare y_true and y_score
    y_true = np.array([1] * len(positives) + [0] * len(negatives))
    y_score = np.array(positives + negatives)

    roc_path = f"similarities/{original_person}"

    try:
        os.makedirs(roc_path, exist_ok=True)
        np.save(os.path.join(roc_path, "y_score.npy"), y_score)
        np.save(os.path.join(roc_path, "y_true.npy"), y_true)
    except OSError as e:
        print("Ova putanja već postoji ili nije moguće kreirati direktorij:", e)
        return
    print("Prosječno vrijeme potrebno za izvlačenje značajki po slici:", np.mean(avg_embedding_time), "sekundi")
    print("Uspješno spremljene vrijednosti y_true i y_score za osobu:", original_person)


def plot_roc(original_person):
    from sklearn.metrics import roc_curve, auc

    y_true = np.load(f"similarities/{original_person}/y_true.npy")
    y_score = np.load(f"similarities/{original_person}/y_score.npy")

    if y_true is None or y_score is None:
        print("Nema spremljenih vrijednosti y_true i y_score za osobu:", original_person)
        return

    # Calculate ROC
    fpr, tpr, thresholds_arr = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title(f'ROC Curve - Face Recognition - {original_person}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Find EER (Equal Error Rate)
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds_arr[eer_idx]
    eer = fpr[eer_idx]
    print(f"EER: {eer:.4f} at threshold ≈ {eer_threshold:.4f}")

    far_threshold = thresholds_arr[np.where(tpr >= 0.01)[0][0]]
    print(f"FAR: 1% at threshold ≈ {far_threshold:.4f}")
    tpr_threshold = thresholds_arr[np.where(tpr >= 0.9)[0][0]]
    print(f"TPR: 90% at threshold ≈ {tpr_threshold:.4f}")


if __name__ == "__main__":
    for person in os.listdir(r"dataset\val"):
        make_labels_and_scores(person)
        plot_roc(person)
        similarity_to_centroid(person)