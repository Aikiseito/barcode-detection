"""Logging utilities for the barcode detection project"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file name (optional)
        log_dir: Directory for log files
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_path = log_path / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce verbosity of some third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class MLflowLogger:
    """Custom logger for MLflow integration."""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: MLflow experiment name
            run_name: MLflow run name (optional)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.logger = get_logger(f"mlflow.{experiment_name}")
        
    def log_params(self, params: dict) -> None:
        """Log parameters to MLflow."""
        import mlflow
        
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
                self.logger.debug(f"Logged parameter: {key}={value}")
            except Exception as e:
                self.logger.warning(f"Failed to log parameter {key}: {e}")
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        import mlflow
        
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
                self.logger.debug(f"Logged metric: {key}={value}")
            except Exception as e:
                self.logger.warning(f"Failed to log metric {key}: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        import mlflow
        
        try:
            if artifact_name:
                mlflow.log_artifact(artifact_path, artifact_name)
            else:
                mlflow.log_artifact(artifact_path)
            self.logger.debug(f"Logged artifact: {artifact_path}")
        except Exception as e:
            self.logger.warning(f"Failed to log artifact {artifact_path}: {e}")
    
    def log_model(self, model, artifact_path: str, **kwargs) -> None:
        """Log model to MLflow."""
        import mlflow.pytorch
        
        try:
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            self.logger.info(f"Logged model to: {artifact_path}")
        except Exception as e:
            self.logger.error(f"Failed to log model: {e}")


def log_system_info() -> None:
    """Log system information."""
    import platform
    import psutil
    import torch
    
    logger = get_logger("system")
    
    # System info
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    # PyTorch info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")


def setup_mlflow_logging(
    tracking_uri: str = "http://127.0.0.1:8080",
    experiment_name: str = "barcode_detection"
) -> None:
    """
    Set up MLflow logging.
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Experiment name
    """
    import mlflow
    
    logger = get_logger("mlflow_setup")
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                mlflow.set_experiment(experiment_name)
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to set experiment: {e}")
            
    except Exception as e:
        logger.error(f"Failed to set up MLflow: {e}")


class TqdmHandler(logging.Handler):
    """Logging handler that works with tqdm progress bars."""
    
    def emit(self, record):
        """Emit a log record."""
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            # Fallback to print if tqdm is not available
            print(self.format(record))


def setup_training_logging(log_dir: str = "logs") -> None:
    """Set up logging specifically for training."""
    from datetime import datetime
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_{timestamp}.log"
    
    setup_logging(
        level="INFO",
        log_file=log_file,
        log_dir=log_dir
    )
    
    # Log system information
    log_system_info()
    
    logger = get_logger("training")
    logger.info("Training logging initialized")


def setup_inference_logging(log_dir: str = "logs") -> None:
    """Set up logging specifically for inference."""
    from datetime import datetime
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"inference_{timestamp}.log"
    
    setup_logging(
        level="INFO",
        log_file=log_file,
        log_dir=log_dir
    )
    
    logger = get_logger("inference")
    logger.info("Inference logging initialized")
