from config import get_config
from core.detector import FaceDetector
from core.model import EmbeddingModel
from core.preprocessor import ImagePreprocessor

class ComponentFactory:
    """Factory that creates configured components from config.yaml"""

    @staticmethod
    def create_model():
        cfg = get_config()
        model_cfg = cfg.models[cfg.backend]
        model = EmbeddingModel(model_cfg.name)
        if model_cfg.quantization.get("enabled", False):
            model.quantize()
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
            target_size=112,
            use_clahe=cfg.preprocessing['use_clahe'],
            use_sharpen=cfg.preprocessing['use_sharpen'],
            use_alignment=cfg.preprocessing['use_alignment']
        )