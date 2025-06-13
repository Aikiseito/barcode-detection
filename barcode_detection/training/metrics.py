# barcode_detection/training/metrics.py
"""
Метрики для обучения модели детекции штрих-кодов.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics import Accuracy, Precision, Recall, F1Score

class BarcodeMetrics:
    """Класс для вычисления метрик детекции штрих-кодов."""
    
    def __init__(self, num_classes: int = 1, device: str = 'cpu'):
        """
        Инициализация метрик.
        
        Args:
            num_classes: Количество классов
            device: Устройство для вычислений
        """
        self.num_classes = num_classes
        self.device = device
        
        # Метрики детекции объектов
        try:
            self.map_metric = MeanAveragePrecision()
        except Exception:
            print("Внимание: MeanAveragePrecision недоступна, используется упрощенная версия")
            self.map_metric = None
        
        # Базовые метрики классификации
        self.accuracy = Accuracy(task='multiclass', num_classes=max(2, num_classes))
        self.precision = Precision(task='multiclass', num_classes=max(2, num_classes), average='macro')
        self.recall = Recall(task='multiclass', num_classes=max(2, num_classes), average='macro')
        self.f1 = F1Score(task='multiclass', num_classes=max(2, num_classes), average='macro')
        
        # Списки для накопления результатов
        self.predictions = []
        self.targets = []
        
    def update(self, predictions: List[Dict], targets: List[Dict]) -> None:
        """
        Обновление метрик новыми предсказаниями.
        
        Args:
            predictions: Список предсказаний модели
            targets: Список истинных меток
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        
        # Обновляем MAP если доступно
        if self.map_metric is not None:
            try:
                self.map_metric.update(predictions, targets)
            except Exception as e:
                print(f"Ошибка обновления MAP: {e}")
    
    def compute(self) -> Dict[str, float]:
        """
        Вычисление всех метрик.
        
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        metrics = {}
        
        # Вычисляем MAP если доступно
        if self.map_metric is not None:
            try:
                map_result = self.map_metric.compute()
                if isinstance(map_result, dict):
                    metrics.update({f"map_{k}": v.item() if hasattr(v, 'item') else v 
                                  for k, v in map_result.items()})
                else:
                    metrics['map'] = map_result.item() if hasattr(map_result, 'item') else map_result
            except Exception as e:
                print(f"Ошибка вычисления MAP: {e}")
                metrics['map'] = 0.0
        
        # Вычисляем IoU для детекции
        if self.predictions and self.targets:
            iou = self._compute_iou()
            metrics['iou'] = iou
        
        # Добавляем простые метрики
        metrics.update({
            'precision': self._compute_simple_precision(),
            'recall': self._compute_simple_recall(),
            'f1_score': self._compute_simple_f1(),
            'accuracy': self._compute_simple_accuracy()
        })
        
        return metrics
    
    def _compute_iou(self) -> float:
        """Вычисляет среднее IoU для всех детекций."""
        if not self.predictions or not self.targets:
            return 0.0
        
        ious = []
        for pred, target in zip(self.predictions, self.targets):
            if 'boxes' in pred and 'boxes' in target:
                pred_boxes = pred['boxes']
                target_boxes = target['boxes']
                
                if len(pred_boxes) > 0 and len(target_boxes) > 0:
                    # Берем первый бокс для простоты
                    iou = self._box_iou(pred_boxes[0], target_boxes[0])
                    ious.append(iou)
        
        return float(np.mean(ious)) if ious else 0.0
    
    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Вычисляет IoU между двумя боксами."""
        # Преобразуем в numpy если нужно
        if isinstance(box1, torch.Tensor):
            box1 = box1.cpu().numpy()
        if isinstance(box2, torch.Tensor):
            box2 = box2.cpu().numpy()
        
        # Вычисляем площади
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Находим пересечение
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_simple_precision(self) -> float:
        """Вычисляет простую точность."""
        if not self.predictions:
            return 0.0
        
        correct = 0
        total = 0
        
        for pred, target in zip(self.predictions, self.targets):
            if 'scores' in pred and len(pred['scores']) > 0:
                # Считаем детекцию правильной если confidence > 0.5
                if pred['scores'][0] > 0.5:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_simple_recall(self) -> float:
        """Вычисляет простую полноту."""
        if not self.targets:
            return 0.0
        
        detected = 0
        total_objects = 0
        
        for pred, target in zip(self.predictions, self.targets):
            if 'boxes' in target:
                total_objects += len(target['boxes'])
                if 'boxes' in pred and len(pred['boxes']) > 0:
                    detected += min(len(pred['boxes']), len(target['boxes']))
        
        return detected / total_objects if total_objects > 0 else 0.0
    
    def _compute_simple_f1(self) -> float:
        """Вычисляет F1-score."""
        precision = self._compute_simple_precision()
        recall = self._compute_simple_recall()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _compute_simple_accuracy(self) -> float:
        """Вычисляет простую точность детекции."""
        if not self.predictions:
            return 0.0
        
        correct = 0
        total = len(self.predictions)
        
        for pred, target in zip(self.predictions, self.targets):
            # Считаем правильным если есть детекция и есть цель
            has_prediction = 'boxes' in pred and len(pred['boxes']) > 0
            has_target = 'boxes' in target and len(target['boxes']) > 0
            
            if has_prediction == has_target:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def reset(self) -> None:
        """Сбрасывает накопленные метрики."""
        self.predictions.clear()
        self.targets.clear()
        
        if self.map_metric is not None:
            self.map_metric.reset()
    
    def get_average_precision(self, iou_threshold: float = 0.5) -> float:
        """
        Вычисляет Average Precision для заданного IoU threshold.
        
        Args:
            iou_threshold: Порог IoU для считывания детекции правильной
            
        Returns:
            float: Average Precision
        """
        if not self.predictions or not self.targets:
            return 0.0
        
        # Простая имплементация AP
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, target in zip(self.predictions, self.targets):
            pred_boxes = pred.get('boxes', [])
            target_boxes = target.get('boxes', [])
            
            if len(target_boxes) == 0:
                false_positives += len(pred_boxes)
            elif len(pred_boxes) == 0:
                false_negatives += len(target_boxes)
            else:
                # Простое сопоставление по IoU
                for pred_box in pred_boxes:
                    best_iou = 0
                    for target_box in target_boxes:
                        iou = self._box_iou(pred_box, target_box)
                        best_iou = max(best_iou, iou)
                    
                    if best_iou >= iou_threshold:
                        true_positives += 1
                    else:
                        false_positives += 1
                
                # Подсчет пропущенных целей
                for target_box in target_boxes:
                    best_iou = 0
                    for pred_box in pred_boxes:
                        iou = self._box_iou(pred_box, target_box)
                        best_iou = max(best_iou, iou)
                    
                    if best_iou < iou_threshold:
                        false_negatives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Упрощенная AP как среднее precision и recall
        return (precision + recall) / 2

def calculate_metrics(predictions, targets):
    """
    Простая функция для расчета метрик детекции штрих-кодов
    
    Args:
        predictions: предсказания модели
        targets: истинные значения
        
    Returns:
        dict: словарь с метриками
    """
    try:
        metrics = BarcodeMetrics()
        metrics.update(predictions, targets)
        results = metrics.compute()
        
        return {
            'map': results.get('map', 0.0),
            'precision': results.get('precision', 0.0),
            'recall': results.get('recall', 0.0),
            'f1': results.get('f1', 0.0)
        }
    except Exception as e:
        print(f"Ошибка при расчете метрик: {e}")
        return {
            'map': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
