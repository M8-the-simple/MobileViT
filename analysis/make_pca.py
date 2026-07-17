import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np
import cv2
from factory import ComponentFactory
from config import get_config

cfg = get_config()
detector = ComponentFactory.create_detector()
preprocessor = ComponentFactory.create_preprocessor()
model = ComponentFactory.create_model()

thresholds = cfg.models[cfg.backend].thresholds

COLORS = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']

def get_embedding_from_image(image_path: str) -> np.ndarray:
    """
    Load image, detect face, preprocess, and extract embedding.
    Returns None if no face detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️  Ne mogu otvoriti sliku: {image_path}")
        return None

    boxes, probs, landmarks = detector.detect(img)

    if len(boxes) == 0:
        print(f"⚠️  Nije detektirano lice: {image_path}")
        return None

    # Take highest confidence face
    best_idx = np.argmax(probs) if len(probs) > 0 else 0
    box = boxes[best_idx]
    land = landmarks[best_idx] if landmarks is not None and len(landmarks) > best_idx else None

    # Preprocess
    processed = preprocessor(img, land)

    if processed is None:
        return None

    # Extract embedding
    return model.embed(processed)

def make_pca(embeddings, class_name, i, n_components=2):
    """Vizualizira embeddinge / centroidе u 2D pomoću PCA"""
    if len(embeddings) == 0:
        print(f"Upozorenje: Nema embeddinga za {class_name}")
        return
    
    embeddings = np.array(embeddings)
    
    # PCA na 2 komponente
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Scatter plot
    plt.scatter(embeddings_2d[:, 0], 
                embeddings_2d[:, 1],
                c=[COLORS[i % len(COLORS)]], 
                s=80, 
                label=class_name,
                edgecolors='black',
                linewidth=0.5)
    
    # Opcionalno: označi centroid svake klase većim krugom ili X-om
    if len(embeddings) > 1:
        class_centroid = embeddings_2d.mean(axis=0)
        plt.scatter(class_centroid[0], class_centroid[1], 
                    c=COLORS[i % len(COLORS)], 
                    s=200, marker='X', edgecolors='black', linewidth=2)

def create_pca_for_centroids():
    centroids_path = os.path.join("centroids", 
                                  model.name.replace(":", "_").replace("/", "_"))
    
    if not os.path.exists(centroids_path):
        print(f"Nema direktorija: {centroids_path}")
        return
    
    embeddings = np.load(os.path.join(centroids_path, "centroid_znacajke.npy"))
    labels = np.load(os.path.join(centroids_path, "oznake.npy"))
    
    print(f"Učitano {embeddings.shape[0]} centroida, dimenzija: {embeddings.shape[1]}")
    
    if embeddings.shape[0] < 2:
        print("Nema dovoljno centroida za PCA.")
        return
    
    # === PCA na SVIM centroidima odjednom ===
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    explained = pca.explained_variance_ratio_
    print(f"Objašnjena varijanca: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}")
    
    plt.figure(figsize=(12, 9))
    
    unique_labels = np.unique(labels)
    for i, class_name in enumerate(unique_labels):
        mask = labels == class_name
        class_2d = embeddings_2d[mask]
        
        # Crtamo točku
        plt.scatter(class_2d[:, 0], class_2d[:, 1],
                    c=[COLORS[i % len(COLORS)]], 
                    s=120, 
                    label=class_name,
                    edgecolors='black',
                    linewidth=1.2)
        
        # Ime osobe pored točke
        plt.annotate(class_name, 
                    (class_2d[0, 0] + 0.01, class_2d[0, 1] + 0.01),
                    fontsize=9)
    
    plt.title('PCA projekcija centroida lica (svi zajedno)', fontsize=14)
    plt.xlabel(f'Glavna komponenta 1 ({explained[0]:.1%})')
    plt.ylabel(f'Glavna komponenta 2 ({explained[1]:.1%})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_pca_for_faces():
    person_path = os.path.join("dataset", "val")
    if not os.path.exists(person_path):
        print(f"Nema direktorija: {person_path}")
        return
    
    plt.figure(figsize=(12, 9))
    
    for i, class_name in enumerate(os.listdir(person_path)):
        class_path = os.path.join(person_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        embeddings = []
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            emb = get_embedding_from_image(img_path)
            if emb is not None:
                embeddings.append(emb)
        
        print(f"Osoba {class_name}: {len(embeddings)} embeddinga")
        make_pca(embeddings, class_name, i)
    
    plt.title('PCA projekcija svih lica iz dataseta', fontsize=14)
    plt.xlabel('Glavna komponenta 1')
    plt.ylabel('Glavna komponenta 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    create_pca_for_faces()
    create_pca_for_centroids()