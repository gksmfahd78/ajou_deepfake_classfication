"""
Models package for deepfake detection system.

This package contains:
- YOLOFaceDetector: YOLO-based face detection model
- DeepfakeClassifier: EfficientNet-based deepfake classification model
"""

from .face_detector import YOLOFaceDetector
from .deepfake_classifier import DeepfakeClassifier

__all__ = [
    'YOLOFaceDetector',
    'DeepfakeClassifier',
]

__version__ = '1.0.0'