# core/model.py
import timm
import torch
import os
from torchvision import transforms
from typing import Tuple
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from torchao.quantization.granularity import PerTensor

class EmbeddingModel:
    def __init__(self, name: str, num_classes: int = 0):
        self.name = name.replace(":", "_").replace("/", "_")
        if name.startswith("buffalo_sc"):
            self.model = FaceAnalysis(name, providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=-1, det_size=(256, 256))
        else:
            self.model = timm.create_model(name, pretrained=True, num_classes=num_classes)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            # Get transform from timm
            data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
            self.transform = timm.data.create_transform(**data_config, is_training=False)

            # Override for huggingface models that expect 112x112
            if name.startswith("hf_hub"):
                self.transform = transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        print(f"[INFO] Model {self.name} initialized on {self.device if not name.startswith('buffalo_sc') else 'CPU'}")

    def quantize(self, mode="int8_wo"):
        if mode == "int8_wo":
            quantize_(self.model, Int8WeightOnlyConfig())
            print("[INFO] Model je kvantiziran na int8")
        else:
            print(f"[INFO] Nepoznat način kvantizacije: {mode}. Model nije kvantiziran.")
        

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from RGB image array."""
        if self.name.startswith("buffalo_sc"):
            faces = self.model.get(image)
            if len(faces) < 1:
                return None 
            return faces[0].normed_embedding.astype(np.float32)
        pil = Image.fromarray(image)
        x = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(x)

        emb_np = emb.cpu().numpy().flatten()
        return emb_np / (np.linalg.norm(emb_np) + 1e-12)
