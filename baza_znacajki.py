import numpy as np
import os
from Embedding_model import get_transform, get_embedding_model
from facenet_pytorch import MTCNN
from utils import get_embedding


def main():
    model, device = get_embedding_model()
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False, select_largest=False, min_face_size=40)
    transform = get_transform()

    face_centroids = {}
    for class_name in os.listdir("dataset/train"):
        class_path = os.path.join("dataset/train", class_name)
        if not os.path.isdir(class_path):
            continue

        embs = []
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            emb = get_embedding(img_path, mtcnn=mtcnn, transform=transform, model=model, device=device)
            if emb is None:
                continue
            print(f"{np.linalg.norm(emb):.4f}")
            embs.append(emb)

        if embs:
            centroid = np.mean(embs, axis=0)
            face_centroids[class_name] = centroid
            print(f"Osoba {class_name}: {len(embs)} slika")

    if not face_centroids:
        print("Nisu pronađeni embeddingi za izračun centroida.")
        return

    print(f"Centoridski vektori: {len(face_centroids)} osoba")

    mean_embs = np.stack(list(face_centroids.values()), axis=0)
    mean_labels = np.array(list(face_centroids.keys()))

    np.save("centroid_znacajke.npy", mean_embs)
    np.save("oznake.npy", mean_labels)


if __name__ == "__main__":
    main()
