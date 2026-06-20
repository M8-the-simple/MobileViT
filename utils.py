import cv2
import numpy as np
from PIL import Image
import torch
import os

from Embedding_model import get_embedding_model, get_transform
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
    Ekstrahiraj normalizirani embedding za lice na slici
    
    Args:
        image_input: path slike (str) ili numpy array (BGR)
        detector: FaceDetector instanca
    """
    # Lazy load defaults
    if model is None or device is None or transform is None or detector is None:
        default_model, default_device, default_transform, default_detector = _get_defaults()
        model = model or default_model
        device = device or default_device
        transform = transform or default_transform
        detector = detector or default_detector

    # Učitaj sliku ako je proslijeđen path
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Ne mogu otvoriti sliku: {image_input}")
    else:
        img = image_input.copy()

    # Detekcija + crop
    face_crop = detect_and_crop(img, min_confidence=0.9, detector=detector)
    
    if face_crop is None or face_crop.size == 0:
        print("Nije detektirano lice na slici")
        return None

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    ## Pokušaj poboljšanja embeddinga
    # Ovdje možeš probati koristiti INTER_LANCZOS4
    # Ovdje možeš probati koristiti INTER_AREA
    face_resized = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    # cv2.imshow("image", face_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Embedding
    pil_img = Image.fromarray(face_resized)
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