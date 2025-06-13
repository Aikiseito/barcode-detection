"""Main commands entry point for barcode detection project"""

import os
import logging
from pathlib import Path
from typing import Optional

import fire
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dvc.api

from barcode_detection.training.trainer import train_model
from barcode_detection.inference.predictor import BarcodePredictor
from barcode_detection.utils.logging import setup_logging
from barcode_detection.utils.helpers import ensure_directories, download_data


def setup_project_environment(config: DictConfig) -> None:
    """Set up project environment and directories"""
    # Set up logging
    setup_logging(config.get('log_level', 'INFO'))
    
    # Create necessary directories
    ensure_directories([
        config.get('output_dir', 'outputs'),
        config.get('plots_dir', 'plots'),
        config.get('logs_dir', 'logs'),
        config.get('data_dir', 'data')
    ])
    
    # Set device
    if config.get('device', 'auto') == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Auto-detected device: {device}")
    else:
        device = config.device
        logging.info(f"Using specified device: {device}")
    
    # Set environment variables
    os.environ['TORCH_DEVICE'] = device


def pull_dvc_data() -> None:
    """Pull data using DVC"""
    try:
        logging.info("Pulling data with DVC...")
        dvc.api.pull()
        logging.info("DVC data pulled successfully")
    except Exception as e:
        logging.error(f"Failed to pull DVC data: {e}")
        logging.info("You may need to manually download the data")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train the barcode detection model."""
    logging.info("Starting training...")
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set up environment
    setup_project_environment(cfg)
    
    # Pull DVC data if needed
    if cfg.data.get('data_source') == 'dvc' or cfg.data.get('pull_dvc', True):
        pull_dvc_data()
    
    # Train model
    model, results = train_model(
        data_config=OmegaConf.to_container(cfg.data, resolve=True),
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        training_config=OmegaConf.to_container(cfg.training, resolve=True),
        mlflow_config=OmegaConf.to_container(cfg.mlflow, resolve=True),
        output_dir=cfg.get('output_dir', 'outputs'),
        experiment_name=cfg.get('experiment_name', 'barcode_detection'),
        seed=cfg.get('seed', 42)
    )
    
    logging.info("Training completed successfully!")
    logging.info(f"Best model path: {results.get('best_model_path')}")
    logging.info(f"Best score: {results.get('best_score')}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def infer(cfg: DictConfig, input_path: str, output_path: str, model_path: Optional[str] = None) -> None:
    """Run inference on images."""
    logging.info("Starting inference...")
    
    # Set up environment
    setup_project_environment(cfg)
    
    # Use provided model path or default from config
    if model_path is None:
        model_path = cfg.inference.get('model_path', 'outputs/best.pt')
    
    # Initialize predictor
    predictor = BarcodePredictor(
        model_path=model_path,
        config=OmegaConf.to_container(cfg.inference, resolve=True),
        device=cfg.get('device', 'auto')
    )
    
    # Run inference
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if input_path.is_file():
        # Single image inference
        results = predictor.predict_image(input_path)
        predictor.save_results(results, output_path)
        logging.info(f"Inference results saved to {output_path}")
    elif input_path.is_dir():
        # Batch inference
        results = predictor.predict_batch(input_path)
        predictor.save_batch_results(results, output_path)
        logging.info(f"Batch inference results saved to {output_path}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluate(cfg: DictConfig, model_path: Optional[str] = None, data_split: str = "test") -> None:
    """Evaluate model on test data"""
    logging.info("Starting evaluation...")
    
    # Set up environment
    setup_project_environment(cfg)
    
    # Use provided model path or default from config
    if model_path is None:
        model_path = cfg.inference.get('model_path', 'outputs/best.pt')
    
    # Initialize predictor
    predictor = BarcodePredictor(
        model_path=model_path,
        config=OmegaConf.to_container(cfg.inference, resolve=True),
        device=cfg.get('device', 'auto')
    )
    
    # Run evaluation
    data_path = Path(cfg.data_dir) / data_split
    metrics = predictor.evaluate(data_path)
    
    logging.info("Evaluation results:")
    for metric_name, metric_value in metrics.items():
        logging.info(f"{metric_name}: {metric_value:.4f}")


def convert_to_onnx(model_path: str, output_path: str, input_size: int = 640) -> None:
    """Convert PyTorch model to ONNX format"""
    import torch
    from barcode_detection.models.yolo_lightning import YOLOLightning
    
    logging.info(f"Converting model {model_path} to ONNX format...")
    
    # Load model
    model = YOLOLightning.load_from_checkpoint(model_path)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model.model.model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logging.info(f"ONNX model saved to {output_path}")


def convert_to_tensorrt(onnx_path: str, output_path: str, precision: str = "fp16") -> None:
    """Convert ONNX model to TensorRT format"""
    try:
        import tensorrt as trt
        from tensorrt import runtime as trt_runtime
    except ImportError:
        raise ImportError("TensorRT is not installed. Please install TensorRT to use this feature.")
    
    logging.info(f"Converting ONNX model {onnx_path} to TensorRT format...")
    
    # Create TensorRT logger and builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
    
    # Set workspace size
    config.max_workspace_size = 1 << 30  # 1 GB
    
    # Create network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                logging.error(f"ONNX parser error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # Build engine
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    logging.info(f"TensorRT engine saved to {output_path}")


def download_sample_data() -> None:
    """Download sample data for testing"""
    logging.info("Downloading sample data...")
    try:
        download_data()
        logging.info("Sample data downloaded successfully")
    except Exception as e:
        logging.error(f"Failed to download sample data: {e}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def serve(cfg: DictConfig, model_path: Optional[str] = None, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start inference server"""
    from barcode_detection.inference.server import start_server
    
    logging.info("Starting inference server...")
    
    # Set up environment
    setup_project_environment(cfg)
    
    # Use provided model path or default from config
    if model_path is None:
        model_path = cfg.inference.get('model_path', 'outputs/best.pt')
    
    # Start server
    start_server(
        model_path=model_path,
        config=OmegaConf.to_container(cfg.inference, resolve=True),
        host=host,
        port=port
    )


class Commands:
    """Main commands class for Fire CLI"""
    
    def train(self, config_name: str = "config", **kwargs):
        """Train the model"""
        # Override config values with command line arguments
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        
        with hydra.initialize(version_base=None, config_path="configs"):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            train(cfg)
    
    def infer(self, input_path: str, output_path: str, model_path: Optional[str] = None, config_name: str = "config"):
        """Run inference"""
        with hydra.initialize(version_base=None, config_path="configs"):
            cfg = hydra.compose(config_name=config_name)
            infer(cfg, input_path, output_path, model_path)
    
    def evaluate(self, model_path: Optional[str] = None, data_split: str = "test", config_name: str = "config"):
        """Evaluate model"""
        with hydra.initialize(version_base=None, config_path="configs"):
            cfg = hydra.compose(config_name=config_name)
            evaluate(cfg, model_path, data_split)
    
    def convert_onnx(self, model_path: str, output_path: str, input_size: int = 640):
        """Convert model to ONNX"""
        convert_to_onnx(model_path, output_path, input_size)
    
    def convert_tensorrt(self, onnx_path: str, output_path: str, precision: str = "fp16"):
        """Convert ONNX to TensorRT"""
        convert_to_tensorrt(onnx_path, output_path, precision)
    
    def serve(self, model_path: Optional[str] = None, host: str = "0.0.0.0", port: int = 8080, config_name: str = "config"):
        """Start inference server"""
        with hydra.initialize(version_base=None, config_path="configs"):
            cfg = hydra.compose(config_name=config_name)
            serve(cfg, model_path, host, port)
    
    def download_data(self):
        """Download sample data"""
        download_sample_data()
    
    def pull_data(self):
        """Pull data using DVC"""
        pull_dvc_data()


def main():
    """Main entry point for Fire CLI"""
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
