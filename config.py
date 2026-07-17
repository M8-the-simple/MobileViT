# pipeline/config.py
import yaml
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Any

_config = None

@dataclass
class ModelConfig:
    name: str
    thresholds: Dict[str, float]
    quantization: Dict[str, bool]  # Add quantization field with default value

@dataclass
class Config:
    backend: str
    frame_skip: int
    temporal_window: int
    detector_method: str
    mtcnn_config: Dict[str, Any]
    haar_config: Dict[str, Any]
    preprocessing: Dict[str, bool]
    models: Dict[str, ModelConfig]

def load_config(path: str = "config.yaml") -> Config:
    global _config
    if _config is None:
        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten structure
        _config = Config(
            backend=data['system']['backend'],
            frame_skip=data['system']['frame_skip'],
            temporal_window=data['system']['temporal_window'],
            detector_method=data['detector']['method'],
            mtcnn_config=data['detector']['mtcnn'],
            haar_config=data['detector']['haar'],
            preprocessing=data['preprocessing'],
            models={
                k: ModelConfig(name=v['name'], thresholds=v['thresholds'], quantization=v['quantization'])
                for k, v in data['models'].items()
            }
        )
    return _config

@lru_cache()
def get_config() -> Config:
    return load_config()