"""
Training modules for deepfake detection system.

This package contains:
- train_yolo_face: YOLO face detector training
- train_deepfake_classifier: EfficientNet classifier training
"""

from .train_yolo_face import train_yolo_face_detector, create_dataset_yaml
from .train_deepfake_classifier import train_deepfake_classifier

__all__ = [
    'train_yolo_face_detector',
    'create_dataset_yaml',
    'train_deepfake_classifier',
]

__version__ = '1.0.0'