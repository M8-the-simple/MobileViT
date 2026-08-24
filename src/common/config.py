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
    input_size: int
    thresholds: Dict[str, float]
    quantization: Dict[str, bool]  # Add quantization field with default value

@dataclass
class Config:
    backend: str
    frame_skip: int
    temporal_window: int
    centroid_path: str
    dataset_path: str
    train_videos_path: str
    test_videos_path: str
    similarities_path: str
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
        haar_cfg = data['detector']['haar']
        if isinstance(haar_cfg['min_size'], list):
            haar_cfg['min_size'] = tuple(haar_cfg['min_size'])
        # Flatten structure
        _config = Config(
            backend=data['system']['backend'],
            frame_skip=data['system']['frame_skip'],
            temporal_window=data['system']['temporal_window'],
            centroid_path=data['system']['centroid_path'],
            dataset_path=data['system']['dataset_path'],
            similarities_path=data['system']['similarities_path'],
            train_videos_path=data['system']['train_videos_path'],
            test_videos_path=data['system']['test_videos_path'],
            detector_method=data['detector']['method'],
            mtcnn_config=data['detector']['mtcnn'],
            haar_config=haar_cfg,
            preprocessing=data['preprocessing'],
            models={
                k: ModelConfig(name=v['name'], thresholds=v['thresholds'], quantization=v['quantization'], input_size=v['input_size'])
                for k, v in data['models'].items()
            }
        )
    return _config

@lru_cache()
def get_config() -> Config:
    return load_config()