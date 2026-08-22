from src.common.config import get_config
from src.core.detector import FaceDetector
from src.core.model import EmbeddingModel
from src.core.preprocessor import ImagePreprocessor

class ComponentFactory:
    """Factory that creates configured components from config.yaml"""

    @staticmethod
    def create_model():
        # import torch
        # def check_quantized(model):
        #     quantized = 0
        #     total = 0
        #     for name, module in model.named_modules():
        #         if isinstance(module, torch.nn.Linear):
        #             total += 1
        #             w = module.weight
        #             t = str(type(w))
        #             if "Quantized" in t or "Int8" in t or "Affine" in t or "int8" in t.lower():
        #                 quantized += 1
        #                 print(f"  [KVANTIZIRANO] {name}: {type(w)}")
        #             else:
        #                 print(f"  [NIJE] {name}: {type(w)} dtype={getattr(w, 'dtype', None)}")
        #     print(f"Kvantizirano Linear slojeva: {quantized}/{total}")
        cfg = get_config()
        model_cfg = cfg.models[cfg.backend]
        model = EmbeddingModel(model_cfg.name)
        # Upotreba:
        #print("=== PRIJE ===")
        #check_quantized(model.model)
        if model_cfg.quantization.get("enabled", False):
             model.quantize()   
        #print("\n=== POSLIJE ===")
        #check_quantized(model.model)
        return model

    @staticmethod
    def create_detector(device: str = "cpu"):
        cfg = get_config()
        if cfg.detector_method == "mtcnn":
            config = cfg.mtcnn_config
        else:
            config = cfg.haar_config
        return FaceDetector(cfg.detector_method, device, config)

    @staticmethod
    def create_preprocessor():
        cfg = get_config()
        return ImagePreprocessor(
            target_size=cfg.models[cfg.backend].input_size,
            use_clahe=cfg.preprocessing['use_clahe'],
            use_sharpen=cfg.preprocessing['use_sharpen'],
            use_alignment=cfg.preprocessing['use_alignment']
        )