from typing import Tuple, Optional
import torch
from detectors import FaceDetector

# Globalni aktivni backend
_active_backend = "buffalo_sc"        # default
_active_model = None
_active_transform = None
_active_device = None
_active_detector = None

def set_active_backend(backend_name: str = "buffalo_sc"):
    """Postavi koji backend koristimo globalno"""
    global _active_backend, _active_model, _active_transform, _active_device, _active_detector
    
    if backend_name == _active_backend and _active_model is not None:
        return  # već postavljeno
    
    _active_backend = backend_name
    
    # Očisti stare reference
    _active_model = None
    _active_detector = None
    _active_transform = None
    
    print(f"[INFO] Aktiviran backend: {backend_name}")


def get_embedding_model(numclasses=0):
    """Vraća trenutno aktivni model (lazy loading)"""
    global _active_model, _active_device, _active_detector
    
    if _active_model is None:
        if _active_backend == "buffalo_sc":
            from .insightface_model import get_insightface_model
            _active_model, _active_device = get_insightface_model(numclasses)
        else:
            from .timm_model import get_timm_model
            _active_model, _active_device = get_timm_model(numclasses, _active_backend)
    
    return _active_model, _active_device


def get_transform():
    global _active_transform
    if _active_transform is None:
        if _active_backend == "buffalo_sc":
            from .insightface_model import get_insightface_transform
            _active_transform = get_insightface_transform()
        else:
            from .timm_model import get_timm_transform
            _active_transform = get_timm_transform()
    return _active_transform

def get_detector():
    global _active_detector
    if(_active_backend == "buffalo_sc"):
        return None
    else:
        if(_active_detector is None):
            _active_detector = FaceDetector(method="mtcnn", device=_active_device)
        return _active_detector
    