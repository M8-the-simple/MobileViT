import cv2
import numpy as np
from PIL import Image
import torch
import os

#from Embedding_model import get_embedding_model, get_transform
from Embedding_model_insightface import get_embedding_model, get_transform
from detectors import FaceDetector


_DEFAULT_MODEL = None
_DEFAULT_DEVICE = None
_DEFAULT_TRANSFORM = None
_DEFAULT_DETECTOR = None


def _get_defaults():
    """Lazy inicijalizacija svih default objekata"""
    global _DEFAULT_MODEL, _DEFAULT_DEVICE, _DEFAULT_TRANSFORM, _DEFAULT_DETECTOR

    if _DEFAULT_MODEL is None or _DEFAULT_DEVICE is None:
        _DEFAULT_MODEL, _DEFAULT_DEVICE = get_embedding_model()
    
    if _DEFAULT_TRANSFORM is None:
        _DEFAULT_TRANSFORM = get_transform()
    
    if _DEFAULT_DETECTOR is None:
        # Ovdje odaberi što želiš koristiti (promijeni po potrebi)
        _DEFAULT_DETECTOR = FaceDetector(method="mtcnn")   # "haar" ili "mtcnn"
    
    return _DEFAULT_MODEL, _DEFAULT_DEVICE, _DEFAULT_TRANSFORM, _DEFAULT_DETECTOR


def detect_and_crop(img: np.ndarray, min_confidence=0.9, detector=None):
    """
    Detektira lice i vraća izrezano lice (BGR).
    Radi i sa Haar i sa MTCNN detektorom.
    """
    if detector is None:
        _, _, _, detector = _get_defaults()

    # Haar cascade nema confidence, pa ga ignoriramo za njega
    face_crop = detector.detect_and_crop(img, min_confidence=min_confidence)
    
    if face_crop is None:
        # print("Nije detektirano lice na slici.")  # zakomentiraj ako smeta u videu
        return None
    
    return face_crop


def get_embedding(image_input, detector=None, transform=None, model=None, device=None):
    """
    Radi i sa starim timm modelima i sa InsightFace-om
    """
    if model is None or device is None or detector is None:
        default_model, default_device, default_transform, default_detector = _get_defaults()
        model = model or default_model
        device = device or default_device
        transform = transform or default_transform

    # Učitaj sliku
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Ne mogu otvoriti sliku: {image_input}")
    else:
        img = image_input.copy()

    # === INSIGHTFACE DETEKCIJA ===
    if hasattr(model, 'app'):           # prepoznajemo InsightFace
        faces = model.app.get(img)
        if len(faces) == 0:
            print("InsightFace: Nije detektirano lice")
            return None
        #print(faces[0].normed_embedding.astype(np.float32))
        return faces[0].normed_embedding.astype(np.float32)

    # === STARI TIMM PUT ===
    # OVDJE TREBA BITI BGR ZBOG TOGA JER U DETECTOR-u se već radi pretvorba iz BGR u RGB
    face_crop = detect_and_crop(img, min_confidence=0.7, detector=default_detector)
    
    if face_crop is None:
        print("Nije detektirano lice na slici")
        return None
    
    pil_img = Image.fromarray(face_crop)
    x = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model(x)
    
    emb_np = emb.cpu().numpy().flatten()
    return emb_np / (np.linalg.norm(emb_np) + 1e-12)


# --- Kompatibilnost / pomoćne funkcije ---
def get_default_detector():
    """Koristno za ostale dijelove koda"""
    _, _, _, detector = _get_defaults()
    return detector

def load_centroids(model):
    """
    Učitava centroidе i oznake za trenutni model.
    Sprema ih u: centroids/{model.name}/
    
    Vraća:
        mean_embs, mean_labels  ili (None, None) ako ne postoje
    """
    if not hasattr(model, 'name') or not model.name:
        model.name = "default_model"   # fallback ako model nema .name

    centroids_dir = os.path.join("centroids", model.name)
    centroid_path = os.path.join(centroids_dir, "centroid_znacajke.npy")
    labels_path   = os.path.join(centroids_dir, "oznake.npy")

    if not os.path.exists(centroid_path) or not os.path.exists(labels_path):
        print(f"⚠️  Centroidi za model '{model.name}' još nisu generirani!")
        print(f"   Očekivana putanja: {centroid_path}")
        return None, None

    mean_embs = np.load(centroid_path)
    mean_labels = np.load(labels_path)
    
    print(f"✅ Učitano {len(mean_labels)} centroida za model: {model.name}")
    return mean_embs, mean_labels