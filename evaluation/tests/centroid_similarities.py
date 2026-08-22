import numpy as np
import os
from collections import Counter
from evaluation.tests import get_embedding_from_image
from src import ComponentFactory
model = ComponentFactory.create_model()
centroids_path = f"centroids/{model.name}"
embeddings = np.load(os.path.join(centroids_path, "centroid_znacajke.npy"))
labels = np.load(os.path.join(centroids_path, "oznake.npy"))

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

    person_path = os.path.join("dataset/test", person)
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
        print(f"{img_name:35} → Maksimalna sličnost prema centroidima: {sim[idx]:.4f} -> {labels_arr[idx]}")

    if len(similarities) != 0:
        print("=" * 60)
        print("Prosječna maksimalna sličnost prema centroidima:", np.mean(similarities))
        print("Prosječna sličnost prema vlastitom centroidu:", np.mean(self_similarities))
        print("Prepoznate osobe prema centroidima:", recognized_people.most_common(3))
        print("=" * 60)

    return similarities, self_similarities

if __name__ == "__main__":

    all_similarities = []
    all_self_similarities = []
    diff = []

    for person in os.listdir("dataset/test"):
        if os.path.isdir(os.path.join("dataset/test", person)):
            s, ss = similarity_to_centroid(person)
            all_similarities.extend(s)
            all_self_similarities.extend(ss)
            diff.append(np.mean(s) - np.mean(ss))
    print(f"Mean similarity to centroids: {np.mean(all_similarities):.4f}")
    print(f"Mean self-similarity: {np.mean(all_self_similarities):.4f}")
    print(f"Biggest difference: {np.max(diff):.4f}")