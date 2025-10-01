"""
Utility functions and classes for deepfake detection system.

This package contains:
- CheckpointManager: Model checkpoint management
- Data utilities: Dataset validation and preprocessing
- Label converter: Custom label format conversion
- Model converter: PyTorch to ONNX conversion
"""

from .checkpoint_manager import CheckpointManager
from .data_utils import validate_dataset, print_dataset_stats
from .label_converter import (
    parse_custom_label,
    convert_to_yolo_format,
    extract_faces_for_classification
)

__all__ = [
    'CheckpointManager',
    'validate_dataset',
    'print_dataset_stats',
    'parse_custom_label',
    'convert_to_yolo_format',
    'extract_faces_for_classification',
]

__version__ = '1.0.0'