# core/model.py
import timm
import torch
from torchvision import transforms
from typing import Tuple
from PIL import Image
import numpy as np
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig    

class EmbeddingModel:
    def __init__(self, name: str, num_classes: int = 0):
        self.name = name
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

    def quantize(self, mode="int8_wo"):
        if mode == "int8_wo":
            quantize_(self.model, Int8WeightOnlyConfig())
            print("[INFO] Model je kvatiziran na int8")
        if mode == "int4_wo":
            quantize_(self.model, Int4WeightOnlyConfig(group_size=32))
            print("[INFO] Model je kvatiziran na int4")

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from RGB image array."""
        pil = Image.fromarray(image)
        x = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(x)

        emb_np = emb.cpu().numpy().flatten()
        return emb_np / (np.linalg.norm(emb_np) + 1e-12)
