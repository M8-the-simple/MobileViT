import numpy as np
import os
import cv2
from PIL import Image
import torch
from torchvision import transforms
from factory import ComponentFactory
from config import get_config



def main(): 

    cfg = get_config()
    model = ComponentFactory.create_model()
    preprocessor = ComponentFactory.create_preprocessor()
    detector = ComponentFactory.create_detector()
    # aug_transform = get_augmentation_transform()
    centroids_dir = os.path.join("centroids", model.name.replace(":", "_").replace("/", "_"))
    
    os.makedirs(centroids_dir, exist_ok=True)

    face_centroids = {}
    for class_name in os.listdir("dataset/train"):
        class_path = os.path.join("dataset", "train", class_name)
    # class_name = "Matija"
    # class_path=rf"C:\Users\matia\Documents\RiTeh\6_semestar\Zavrsni_rad\HaarCascade\MobileViT\dataset\train\Matija"

        if not os.path.isdir(class_path):
            print(f"Folder nije pronađen: {class_path}")
            return

        image_paths = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

        print(f"Osoba {class_name}: {len(image_paths)} originalnih slika")

        embs = []

        for i, img_path in enumerate(image_paths):
            # === ORIGINAL ===
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Original {i+1:2d}: Greška pri učitavanju slike {img_path}")
                continue
            boxes, probs, landmarks = detector.detect(img)
            faces = [(b, p, l) for b, p, l in zip(boxes, probs, landmarks)]
            box, prob, land = max(faces, key=lambda x: x[1] if x[1] is not None else 0) if faces else (None, None, None)
            if box is None:
                print(f"  Original {i+1:2d}: Nije detektirano lice na slici {img_path}")
                continue
            for box, prob, land in zip(boxes, probs, landmarks):
                if prob < 0.5:
                    continue
                preprocessed_img = preprocessor(img, landmarks=land)
                emb = model.embed(preprocessed_img)
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

        centroid_path = os.path.join(centroids_dir, "centroid_znacajke.npy")
        label_path = os.path.join(centroids_dir, "oznake.npy")

        np.save(centroid_path, mean_embs)
        np.save(label_path, mean_labels)
        
        print(f"Centroid za {class_name} uspješno spremljen ({len(embs)} embeddinga) na {centroid_path}")


if __name__ == "__main__":
    main()