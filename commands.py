"""Исправленный основной файл команд для проекта детекции штрих-кодов"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import fire

# Настройка базового логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_imports():
    """Настройка импортов с обработкой ошибок"""
    # Добавляем текущую директорию в путь
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    try:
        # Пытаемся импортировать все необходимые модули
        global train_model, BarcodePredictor, setup_logging, ensure_directories
        global YOLOLightning, BarcodeDataModule
        
        from barcode_detection.training.trainer import train_model
        from barcode_detection.inference.predictor import BarcodePredictor
        from barcode_detection.utils.logging import setup_logging
        from barcode_detection.utils.helpers import ensure_directories
        from barcode_detection.models.yolo_lightning import YOLOLightning
        from barcode_detection.data.data_module import BarcodeDataModule
        
        logger.info("Все модули успешно импортированы")
        return True
        
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        logger.error("Проверьте структуру проекта и убедитесь что все файлы на месте")
        return False

def train(
    config_file: str = "config.yaml",
    data_dir: str = "data",
    output_dir: str = "outputs",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    model_name: str = "yolov8n.pt",
    num_classes: int = 5
):
    """
    Обучение модели детекции штрих-кодов
    
    Args:
        config_file: Путь к файлу конфигурации
        data_dir: Директория с данными
        output_dir: Директория для сохранения результатов
        epochs: Количество эпох обучения
        batch_size: Размер батча
        learning_rate: Скорость обучения
        model_name: Имя модели YOLO
        num_classes: Количество классов
    """
    
    if not setup_imports():
        return
    
    logger.info("Начинаем обучение модели...")
    
    # Создание необходимых директорий
    ensure_directories([output_dir, "logs", "plots"])
    
    # Настройка логирования
    setup_logging(level="INFO", log_dir="logs")
    
    # Конфигурация для обучения
    data_config = {
        'data_dir': data_dir,
        'train_path': 'train',
        'val_path': 'val', 
        'test_path': 'test',
        'image_size': 640,
        'batch_size': batch_size
    }
    
    model_config = {
        'name': model_name,
        'num_classes': num_classes
    }
    
    training_config = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_workers': 4,
        'mixed_precision': True,
        'early_stopping': True,
        'patience': 10
    }
    
    mlflow_config = {
        'tracking_uri': 'http://127.0.0.1:8080',
        'experiment_name': 'barcode_detection'
    }
    
    try:
        # Запуск обучения
        model, results = train_model(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            mlflow_config=mlflow_config,
            output_dir=output_dir
        )
        
        logger.info("Обучение завершено успешно!")
        logger.info(f"Результаты: {results}")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        raise

def infer(
    input_path: str,
    output_path: str,
    model_path: str = "outputs/best.pt",
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45
):
    """
    Запуск инференса на изображениях
    
    Args:
        input_path: Путь к входному изображению или директории
        output_path: Путь для сохранения результатов
        model_path: Путь к обученной модели
        confidence_threshold: Порог уверенности
        iou_threshold: Порог IoU для NMS
    """
    
    if not setup_imports():
        return
        
    logger.info(f"Запуск инференса для: {input_path}")
    
    # Конфигурация для инференса
    config = {
        'confidence_threshold': confidence_threshold,
        'iou_threshold': iou_threshold,
        'max_detections': 100,
        'input_size': 640
    }
    
    try:
        # Инициализация предиктора
        predictor = BarcodePredictor(
            model_path=model_path,
            config=config,
            device="auto"
        )
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if input_path.is_file():
            # Одно изображение
            results = predictor.predict_image(input_path)
            predictor.save_results(results, output_path)
            logger.info(f"Результаты сохранены в: {output_path}")
            
        elif input_path.is_dir():
            # Batch инференс
            results = predictor.predict_batch(input_path)
            predictor.save_batch_results(results, output_path)
            logger.info(f"Результаты batch инференса сохранены в: {output_path}")
            
        else:
            raise ValueError(f"Путь не существует: {input_path}")
            
    except Exception as e:
        logger.error(f"Ошибка при инференсе: {e}")
        raise

def evaluate(
    model_path: str = "outputs/best.pt",
    data_path: str = "data/test",
    output_file: str = "evaluation_results.json"
):
    """
    Оценка качества модели на тестовых данных
    
    Args:
        model_path: Путь к модели
        data_path: Путь к тестовым данным
        output_file: Файл для сохранения результатов оценки
    """
    
    if not setup_imports():
        return
        
    logger.info("Запуск оценки модели...")
    
    config = {
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'max_detections': 100,
        'input_size': 640
    }
    
    try:
        # Инициализация предиктора
        predictor = BarcodePredictor(
            model_path=model_path,
            config=config,
            device="auto"
        )
        
        # Оценка
        metrics = predictor.evaluate(data_path)
        
        # Сохранение результатов
        import json
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info("Результаты оценки:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
            
        logger.info(f"Результаты сохранены в: {output_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при оценке: {e}")
        raise

def serve(
    model_path: str = "outputs/best.pt",
    host: str = "0.0.0.0",
    port: int = 8080
):
    """
    Запуск FastAPI сервера для инференса
    
    Args:
        model_path: Путь к модели
        host: Хост сервера
        port: Порт сервера
    """
    
    logger.info(f"Запуск inference сервера на {host}:{port}")
    
    try:
        # Импорт server модуля
        from server import start_server
        
        config = {
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'input_size': 640
        }
        
        start_server(
            model_path=model_path,
            config=config,
            host=host,
            port=port
        )
        
    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {e}")
        raise

def setup_project():
    """Настройка структуры проекта"""
    
    logger.info("Настройка структуры проекта...")
    
    # Создание необходимых директорий
    directories = [
        "data/train", "data/val", "data/test",
        "outputs", "logs", "plots", "configs",
        "barcode_detection/data",
        "barcode_detection/models", 
        "barcode_detection/training",
        "barcode_detection/inference",
        "barcode_detection/utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Создана директория: {directory}")
    
    # Создание __init__.py файлов
    init_files = [
        "barcode_detection/__init__.py",
        "barcode_detection/data/__init__.py",
        "barcode_detection/models/__init__.py",
        "barcode_detection/training/__init__.py", 
        "barcode_detection/inference/__init__.py",
        "barcode_detection/utils/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        logger.info(f"Создан файл: {init_file}")
    
    logger.info("Структура проекта настроена успешно!")

class Commands:
    """Основной класс команд для Fire CLI"""
    
    def train(self, **kwargs):
        """Обучение модели"""
        return train(**kwargs)
    
    def infer(self, input_path: str, output_path: str, **kwargs):
        """Инференс"""
        return infer(input_path, output_path, **kwargs)
    
    def evaluate(self, **kwargs):
        """Оценка модели"""
        return evaluate(**kwargs)
    
    def serve(self, **kwargs):
        """Запуск сервера"""
        return serve(**kwargs)
    
    def setup(self):
        """Настройка проекта"""
        return setup_project()

def main():
    """Основная точка входа"""
    try:
        fire.Fire(Commands)
    except Exception as e:
        logger.error(f"Ошибка выполнения команды: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
