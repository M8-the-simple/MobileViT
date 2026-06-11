import cv2
from facenet_pytorch import MTCNN
from Embedding_model import get_embedding_model, get_transform
import numpy as np
from PIL import Image
import torch
import os


_DEFAULT_MODEL = None
_DEFAULT_DEVICE = None
_DEFAULT_TRANSFORM = None
_DEFAULT_MTCNN = None


def _get_defaults():
    global _DEFAULT_MODEL, _DEFAULT_DEVICE, _DEFAULT_TRANSFORM, _DEFAULT_MTCNN

    if _DEFAULT_MODEL is None or _DEFAULT_DEVICE is None:
        _DEFAULT_MODEL, _DEFAULT_DEVICE = get_embedding_model()
    if _DEFAULT_TRANSFORM is None:
        _DEFAULT_TRANSFORM = get_transform()
    if _DEFAULT_MTCNN is None:
        _DEFAULT_MTCNN = MTCNN(keep_all=True, device=_DEFAULT_DEVICE)

    return _DEFAULT_MODEL, _DEFAULT_DEVICE, _DEFAULT_TRANSFORM, _DEFAULT_MTCNN

def detect_and_crop(img, min_confidence=0.7, mtcnn=None):
    if mtcnn is None:
        _, _, _, mtcnn = _get_defaults()

    boxes, probs = mtcnn.detect(img)
    if boxes is not None and probs is not None:
                
                valid_faces = [(box, prob) for box, prob in zip(boxes, probs) 
                  if prob is not None and float(prob) > float(min_confidence)]
            
                if not valid_faces:
                    print("Nije detektirano lice s dovoljnom sigurnošću.")
                    return None
            
                best_box, best_prob = max(valid_faces, key=lambda x: x[1])

                x1, y1, x2, y2 = best_box.astype(int)

                # sigurnosna rezanja da ne izađemo iz slike
                h_frame, w_frame, _ = img.shape

                margin_h = int(0.2 * (y2 - y1)) 
                margin_w = int(0.2 * (x2 - x1))  

                x1 = max(0, x1 - margin_w)
                y1 = max(0, y1 - margin_h)
                x2 = min(w_frame, x2 + margin_w)
                y2 = min(h_frame, y2 + margin_h)

                face_crop = img[y1:y2, x1:x2]  # i dalje BGR, što je ok za get_embedding_from_crop
                return face_crop
    print("Nije detektirano lice na slici.")
    return None  # Ako nema detekcije, vrati original


def get_embedding(image_input, mtcnn=None, transform=None, model=None, device=None):
    """
    Ekstrahiraj normalizirani embedding za lice na slici
    
    Args:
        image_input: path slike ili numpy array (BGR)
    """
    # Učitaj modele ako nisu proslijeđeni
    if model is None or device is None or mtcnn is None or transform is None:
        default_model, default_device, default_transform, default_mtcnn = _get_defaults()
        if model is None:
            model = default_model
        if device is None:
            device = default_device
        if transform is None:
            transform = default_transform
        if mtcnn is None:
            mtcnn = default_mtcnn
    if mtcnn is None:
        mtcnn = MTCNN(keep_all=True, device=device)
    if transform is None:
        transform = get_transform()
    
    # Učitaj sliku
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Ne mogu otvoriti sliku: {image_input}")
    else:
        img = image_input.copy()
    
    # Detektiraj i izreži lice
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_crop = detect_and_crop(img_rgb, min_confidence=0.7, mtcnn=mtcnn)
    
    if face_crop is None:
        print("Nije detektirano lice na slici")
        return None
    
    # Ekstrahiraj embedding
    pil_img = Image.fromarray(face_crop)
    x = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model(x)
    
    # Normaliziraj
    emb_np = emb.cpu().numpy().flatten()
    return emb_np / (np.linalg.norm(emb_np) + 1e-12)  # Dodaj mali epsilon