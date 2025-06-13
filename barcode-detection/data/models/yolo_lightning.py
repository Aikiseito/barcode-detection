import lightning as L
import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Any, Dict, List, Optional
import mlflow
from pathlib import Path

class YOLOLightning(L.LightningModule):
    """PyTorch Lightning обертка для YOLO модели детекции штрих-кодов"""
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        num_classes: int = 8,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Инициализация YOLO модели
        self.model = YOLO(model_name)
        
        # Гиперпараметры
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        # Метрики для отслеживания
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x):
        """Forward pass через YOLO модель"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Шаг обучения"""
        images, targets = batch
        
        # YOLO вычисляет лосс внутренне
        results = self.model.train_step(images, targets)
        loss = results['loss']
        
        # Логирование метрик
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/box_loss', results.get('box_loss', 0), on_epoch=True)
        self.log('train/cls_loss', results.get('cls_loss', 0), on_epoch=True)
        self.log('train/dfl_loss', results.get('dfl_loss', 0), on_epoch=True)
        
        # MLflow логирование
        if mlflow.active_run():
            mlflow.log_metrics({
                'train_loss': loss.item(),
                'train_box_loss': results.get('box_loss', 0),
                'train_cls_loss': resul
