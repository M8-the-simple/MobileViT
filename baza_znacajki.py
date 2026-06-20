import numpy as np
import os
import cv2
from PIL import Image
import torch
from torchvision import transforms
from Embedding_model import get_embedding_model 

from utils import get_embedding
from detectors import FaceDetector   # pretpostavljam da imaš ovu klasu


def get_augmentation_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.4),
    ])


def main():
    #model, device = get_embedding_model()           # ako ti treba
    detector = FaceDetector(method="mtcnn")

    aug_transform = get_augmentation_transform()

    face_centroids = {}
    class_name = "Mathias"
    class_path = os.path.join("dataset", "train", class_name)

    if not os.path.isdir(class_path):
        print(f"Folder nije pronađen: {class_path}")
        return

    image_paths = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

    print(f"Osoba {class_name}: {len(image_paths)} originalnih slika")

    embs = []

    for i, img_path in enumerate(image_paths):
        # === ORIGINAL ===
        emb = get_embedding(img_path, detector=detector)
        if emb is not None:
            embs.append(emb)
            print(f"  Original {i+1:2d}: OK")
        else:
            print(f"  Original {i+1:2d}: nije detektirano lice")

        # === AUGMENTACIJE (5 po slici) ===
        # try:
        #     img_pil = Image.open(img_path).convert('RGB')
            
        #     for aug_idx in range(5):
        #         aug_pil = aug_transform(img_pil)
        #         aug_np = np.array(aug_pil)                    # ovo je RGB
                
        #         # get_embedding obično očekuje BGR ili radi konverziju sam
        #         emb_aug = get_embedding(aug_np, detector=detector)
                
        #         if emb_aug is not None:
        #             embs.append(emb_aug)
        # except Exception as e:
        #     print(f"  Greška pri augmentaciji {img_path}: {e}")

    print(f"\nUkupno generirano embeddinga: {len(embs)}")

    if len(embs) < 10:
        print("UPOZORENJE: Premalo embeddinga za dobar centroid!")
        return

    # Bolji centroid (manje osjetljiv na outliere)
    embs = np.array(embs)
    centroid = np.median(embs, axis=0)          # ← median je bolji od mean-a
    
    face_centroids[class_name] = centroid

    # Spremanje
    mean_embs = np.stack(list(face_centroids.values()), axis=0)
    mean_labels = np.array(list(face_centroids.keys()))

    np.save("centroid_znacajke.npy", mean_embs)
    np.save("oznake.npy", mean_labels)
    
    print(f"Centroid za {class_name} uspješno spremljen ({len(embs)} embeddinga)")


if __name__ == "__main__":
    main()