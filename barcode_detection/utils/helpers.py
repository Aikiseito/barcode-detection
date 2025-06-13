# barcode_detection/utils/helpers.py
"""
Вспомогательные функции для MLOps проекта детекции штрих-кодов.
"""

import os
import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Настройка логирования для проекта.
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу логов (опционально)
        format_string: Формат сообщений логов
    
    Returns:
        logging.Logger: Настроенный логгер
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Создаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Создаем форматтер
    formatter = logging.Formatter(format_string)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый обработчик (если указан файл)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def set_random_seed(seed: int = 42) -> None:
    """
    Устанавливает одинаковое начальное состояние для всех генераторов случайных чисел.
    
    Args:
        seed: Значение seed для воспроизводимости
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir_exists(directory: str) -> Path:
    """
    Создает директорию, если она не существует.
    
    Args:
        directory: Путь к директории
        
    Returns:
        Path: Объект Path для созданной директории
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_device() -> torch.device:
    """
    Определяет доступное устройство для PyTorch.
    
    Returns:
        torch.device: Доступное устройство (CUDA или CPU)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Сохраняет конфигурацию в файл.
    
    Args:
        config: Словарь с конфигурацией
        filepath: Путь к файлу для сохранения
    """
    import json
    
    ensure_dir_exists(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Загружает конфигурацию из файла.
    
    Args:
        filepath: Путь к файлу конфигурации
        
    Returns:
        Dict[str, Any]: Словарь с конфигурацией
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)

def get_git_commit() -> Optional[str]:
    """
    Получает хеш текущего коммита Git.
    
    Returns:
        Optional[str]: Хеш коммита или None если не Git репозиторий
    """
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def format_bytes(bytes_value: int) -> str:
    """
    Форматирует размер в байтах в человеко-читаемый формат.
    
    Args:
        bytes_value: Размер в байтах
        
    Returns:
        str: Отформатированная строка
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} TB"

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Подсчитывает количество параметров в модели.
    
    Args:
        model: PyTorch модель
        
    Returns:
        Dict[str, int]: Словарь с количеством параметров
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def create_dirs_for_project() -> None:
    """
    Создает все необходимые директории для проекта.
    """
    dirs = [
        'logs',
        'outputs',
        'plots',
        'mlruns',
        'data/train',
        'data/val', 
        'data/test',
        'configs/data',
        'configs/model',
        'configs/training',
        'configs/inference'
    ]
    
    for dir_path in dirs:
        ensure_dir_exists(dir_path)
        print(f"✓ Создана директория: {dir_path}")

# Алиасы для совместимости
setup_logger = setup_logging  # Альтернативное имя

def ensure_directories(config=None):
    """
    Создает необходимые директории для проекта если они не существуют
    
    Args:
        config: опциональный конфигурационный объект (игнорируется для совместимости)
    """
    import os
    
    directories = [
        'logs',
        'outputs', 
        'plots',
        'mlruns',
        'data/train',
        'data/val', 
        'data/test'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Директория {directory} создана/проверена")
        except Exception as e:
            print(f"Предупреждение: не удалось создать директорию {directory}: {e}")
