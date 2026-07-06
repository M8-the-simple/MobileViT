import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import torch

class InsightFaceWrapper:
    def __init__(self, name='buffalo_sc', providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.app = FaceAnalysis(name=name, providers=providers)
        # det_size možeš prilagoditi (veći = bolja točnost, sporije)
        self.app.prepare(ctx_id=0 if 'CUDA' in providers[0] else -1, 
                        det_size=(640, 640))
        self.embedding_dim = 512  # buffalo_sc

    def __call__(self, img_bgr):
        """img_bgr: numpy array u BGR formatu"""
        if isinstance(img_bgr, str):
            img_bgr = cv2.imread(img_bgr)
        
        faces = self.app.get(img_bgr)
        if len(faces) == 0:
            print("InsightFace: Lice nije detektirano")
            return None
        return faces[0].normed_embedding.astype(np.float32)  # već normaliziran

def get_insightface_model(numclasses=0, name: str = "buffalo_sc"):
    """Sada vraća wrapper umjesto timm modela"""
    model = InsightFaceWrapper(name)
    model.name = name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # za kompatibilnost
    return model, device

def get_insightface_transform():
    """InsightFace ne treba transform kao timm — ali da ne razbiješ ostali kod"""
    import torchvision.transforms as T
    # Minimalan transform koji radi s tvojim get_embedding funkcijama
    return T.Compose([
        T.ToTensor(),
    ])