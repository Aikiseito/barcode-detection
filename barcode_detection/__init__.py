"""Barcode Detection MLOps Package"""

__version__ = "0.1.0"
__author__ = "seito"
__description__ = "MLOps barcode and QR code detection system using YOLO deep learning models"

# Основные импорты для удобства использования
from .models.yolo_lightning import YOLOLightning
from .data.dataset import BarcodeDataset
from .data.data_module import BarcodeDataModule
from .training.trainer import train_model, BarcodeTrainer
from .inference.predictor import BarcodePredictor
from .utils.logging import setup_logging
from .utils.helpers import ensure_directories

__all__ = [
    "YOLOLightning",
    "BarcodeDataset", 
    "BarcodeDataModule",
    "train_model",
    "BarcodeTrainer",
    "BarcodePredictor",
    "setup_logging",
    "ensure_directories"
]
