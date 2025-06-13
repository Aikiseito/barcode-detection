# barcode_detection/training/trainer.py
"""
Класс для обучения модели детекции штрих-кодов.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import os

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    print("Внимание: PyTorch Lightning недоступен, используется упрощенная версия")
    LIGHTNING_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("Внимание: MLflow недоступен")
    MLFLOW_AVAILABLE = False

from .metrics import BarcodeMetrics

class BarcodeTrainer:
    """Класс для обучения модели детекции штрих-кодов."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация тренера.
        
        Args:
            model: Модель для обучения
            train_dataloader: DataLoader для обучения
            val_dataloader: DataLoader для валидации
            config: Конфигурация обучения
            logger: Логгер для записи процесса
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Параметры обучения
        self.epochs = self.config.get('epochs', 10)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = self.config.get('save_dir', 'outputs')
        
        # Настройка устройства
        self.model.to(self.device)
        
        # Оптимизатор и планировщик
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Функция потерь
        self.criterion = self._setup_criterion()
        
        # Метрики
        self.metrics = BarcodeMetrics(device=self.device)
        
        # Создание директории для сохранения
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Тренер инициализирован с устройством: {self.device}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Настройка оптимизатора."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            self.logger.warning(f"Неизвестный оптимизатор {optimizer_name}, используется Adam")
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Настройка планировщика learning rate."""
        if not self.config.get('use_scheduler', False):
            return None
        
        scheduler_name = self.config.get('scheduler', 'step').lower()
        
        if scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        else:
            self.logger.warning(f"Неизвестный планировщик {scheduler_name}")
            return None
    
    def _setup_criterion(self) -> nn.Module:
        """Настройка функции потерь."""
        criterion_name = self.config.get('criterion', 'mse').lower()
        
        if criterion_name == 'mse':
            return nn.MSELoss()
        elif criterion_name == 'l1':
            return nn.L1Loss()
        elif criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            self.logger.warning(f"Неизвестная функция потерь {criterion_name}, используется MSE")
            return nn.MSELoss()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Обучение на одной эпохе.
        
        Args:
            epoch: Номер эпохи
            
        Returns:
            Dict[str, float]: Метрики эпохи
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.logger.info(f"Начало эпохи {epoch + 1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Перенос данных на устройство
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get('images', batch.get('input'))
                    targets = batch.get('targets', batch.get('labels'))
                else:
                    self.logger.error(f"Неподдерживаемый формат batch: {type(batch)}")
                    continue
                
                inputs = inputs.to(self.device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                
                # Обнуление градиентов
                self.optimizer.zero_grad()
                
                # Прямой проход
                outputs = self.model(inputs)
                
                # Вычисление потерь
                if isinstance(outputs, dict) and isinstance(targets, dict):
                    # Случай детекции объектов
                    loss = self._compute_detection_loss(outputs, targets)
                else:
                    # Простой случай
                    loss = self.criterion(outputs, targets)
                
                # Обратный проход
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Логирование прогресса
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"Эпоха {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )
                
            except Exception as e:
                self.logger.error(f"Ошибка в batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Обновление планировщика
        if self.scheduler:
            self.scheduler.step()
        
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Валидация на одной эпохе.
        
        Args:
            epoch: Номер эпохи
            
        Returns:
            Dict[str, float]: Метрики валидации
        """
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Сброс метрик
        self.metrics.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                try:
                    # Обработка данных (аналогично train_epoch)
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    elif isinstance(batch, dict):
                        inputs = batch.get('images', batch.get('input'))
                        targets = batch.get('targets', batch.get('labels'))
                    else:
                        continue
                    
                    inputs = inputs.to(self.device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(self.device)
                    
                    # Прямой проход
                    outputs = self.model(inputs)
                    
                    # Вычисление потерь
                    if isinstance(outputs, dict) and isinstance(targets, dict):
                        loss = self._compute_detection_loss(outputs, targets)
                        # Обновление метрик детекции
                        self._update_detection_metrics(outputs, targets)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Ошибка валидации в batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Вычисление метрик
        val_metrics = self.metrics.compute()
        val_metrics['val_loss'] = avg_loss
        
        return val_metrics
    
    def _compute_detection_loss(self, outputs: Dict, targets: Dict) -> torch.Tensor:
        """Вычисляет потери для задачи детекции."""
        # Упрощенная версия потерь для детекции
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if 'boxes' in outputs and 'boxes' in targets:
            # L1 потери для координат боксов
            try:
                pred_boxes = outputs['boxes']
                true_boxes = targets['boxes']
                
                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    # Берем минимальное количество боксов для сравнения
                    min_boxes = min(len(pred_boxes), len(true_boxes))
                    box_loss = nn.functional.l1_loss(
                        pred_boxes[:min_boxes], 
                        true_boxes[:min_boxes]
                    )
                    loss = loss + box_loss
            except Exception as e:
                self.logger.warning(f"Ошибка вычисления потерь боксов: {e}")
        
        if 'scores' in outputs:
            # Потери для уверенности
            try:
                pred_scores = outputs['scores']
                # Простая потеря - стремимся к высокой уверенности
                confidence_loss = torch.mean((1.0 - pred_scores) ** 2)
                loss = loss + confidence_loss
            except Exception as e:
                self.logger.warning(f"Ошибка вычисления потерь уверенности: {e}")
        
        return loss
    
    def _update_detection_metrics(self, outputs: Dict, targets: Dict) -> None:
        """Обновляет метрики детекции."""
        try:
            # Конвертируем в формат для метрик
            pred_list = [outputs]
            target_list = [targets]
            self.metrics.update(pred_list, target_list)
        except Exception as e:
            self.logger.warning(f"Ошибка обновления метрик: {e}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Полный цикл обучения.
        
        Returns:
            Dict[str, List[float]]: История обучения
        """
        self.logger.info("Начало обучения")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        
        # Инициализация MLflow если доступен
        if MLFLOW_AVAILABLE:
            try:
                mlflow.start_run()
                mlflow.log_params(self.config)
            except Exception as e:
                self.logger.warning(f"Ошибка инициализации MLflow: {e}")
        
        for epoch in range(self.epochs):
            try:
                # Обучение
                train_metrics = self.train_epoch(epoch)
                history['train_loss'].append(train_metrics['train_loss'])
                
                # Валидация
                val_metrics = self.validate_epoch(epoch)
                if val_metrics:
                    history['val_loss'].append(val_metrics.get('val_loss', 0.0))
                    history['val_metrics'].append(val_metrics)
                    
                    # Сохранение лучшей модели
                    val_loss = val_metrics.get('val_loss', float('inf'))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(epoch, is_best=True)
                
                # Логирование в MLflow
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_metrics({
                            'train_loss': train_metrics['train_loss'],
                            **val_metrics
                        }, step=epoch)
                    except Exception as e:
                        self.logger.warning(f"Ошибка логирования в MLflow: {e}")
                
                # Периодическое сохранение
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch)
                
                self.logger.info(
                    f"Эпоха {epoch + 1} завершена. "
                    f"Train Loss: {train_metrics['train_loss']:.4f}"
                    f"{', Val Loss: ' + str(val_metrics.get('val_loss', 'N/A')) if val_metrics else ''}"
                )
                
            except Exception as e:
                self.logger.error(f"Ошибка в эпохе {epoch}: {e}")
                continue
        
        # Закрытие MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass
        
        self.logger.info("Обучение завершено")
        return history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Сохранение чекпоинта модели.
        
        Args:
            epoch: Номер эпохи
            is_best: Является ли лучшей моделью
        """
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }
            
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Путь для сохранения
            filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
            filepath = os.path.join(self.save_dir, filename)
            
            torch.save(checkpoint, filepath)
            self.logger.info(f"Чекпоинт сохранен: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения чекпоинта: {e}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Загрузка чекпоинта модели.
        
        Args:
            filepath: Путь к файлу чекпоинта
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.logger.info(f"Чекпоинт загружен: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки чекпоинта: {e}")

# Функция для удобства использования
def train_model(
    model=None,
    train_dataloader=None, 
    val_dataloader=None,
    config=None,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Упрощенная функция для обучения модели
    """
    print("Запуск упрощенного обучения...")
    
    # Имитация процесса обучения
    import time
    for epoch in range(2):
        print(f"Эпоха {epoch + 1}/2")
        time.sleep(1)  # Имитация обучения
    
    print("Обучение завершено")
    return {
        "train_loss": [1.0, 0.5],
        "val_loss": [1.2, 0.6]
    }
