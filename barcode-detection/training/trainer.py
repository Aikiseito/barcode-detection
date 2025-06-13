"""Training utilities for barcode detection"""

import os
import git
import logging
import torch
import pytorch_lightning as pl
import mlflow
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from barcode_detection.models.yolo_lightning import YOLOLightning
from barcode_detection.data.data_module import BarcodeDataModule


def train_model(
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    mlflow_config: Dict[str, Any],
    output_dir: str = "outputs",
    experiment_name: str = "barcode_detection",
    run_name: Optional[str] = None,
    pretrained_weights: Optional[str] = None,
    seed: int = 42
) -> Tuple[pl.LightningModule, Dict[str, Any]]:
    """
    Train a YOLO model for barcode detection.
    
    Args:
        data_config: Data configuration
        model_config: Model configuration
        training_config: Training configuration
        mlflow_config: MLflow configuration
        output_dir: Directory to save outputs
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        pretrained_weights: Path to pretrained weights
        seed: Random seed
        
    Returns:
        Tuple of (trained_model, training_results)
    """
    # Set random seed for reproducibility
    pl.seed_everything(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    data_module = BarcodeDataModule(
        data_config=data_config,
        batch_size=training_config.get('dataloader', {}).get('batch_size', 16),
        num_workers=training_config.get('dataloader', {}).get('num_workers', 4),
        pin_memory=training_config.get('dataloader', {}).get('pin_memory', True)
    )
    
    # Initialize model
    if pretrained_weights and Path(pretrained_weights).exists():
        model = YOLOLightning.load_from_checkpoint(
            pretrained_weights,
            model_config=model_config,
            training_config=training_config,
            num_classes=data_module.get_num_classes()
        )
        logging.info(f"Loaded model from checkpoint: {pretrained_weights}")
    else:
        model = YOLOLightning(
            model_config=model_config,
            training_config=training_config,
            num_classes=data_module.get_num_classes(),
            pretrained_weights=model_config.get('pretrained_weights', 'yolov5s.pt')
        )
        logging.info("Initialized new model")
    
    # Set up MLflow
    setup_mlflow(mlflow_config)
    
    # Configure logger
    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=mlflow_config.get('tracking_uri', "http://127.0.0.1:8080")
    )
    
    # Log git info if available
    if mlflow_config.get('log_git_info', True):
        log_git_info()
    
    # Log parameters
    for param_name in mlflow_config.get('log_params', []):
        if param_name in training_config:
            mlflow.log_param(param_name, training_config[param_name])
        elif '.' in param_name:
            # Handle nested parameters
            parts = param_name.split('.')
            current_dict = training_config
            for part in parts[:-1]:
                if part in current_dict:
                    current_dict = current_dict[part]
                else:
                    current_dict = None
                    break
            if current_dict and parts[-1] in current_dict:
                mlflow.log_param(param_name, current_dict[parts[-1]])
    
    # Configure callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename='{epoch}-{val_map50:.3f}',
        save_top_k=training_config.get('save_top_k', 3),
        monitor=training_config.get('monitor', 'val_map50'),
        mode=training_config.get('mode', 'max'),
        save_last=training_config.get('save_last', True)
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if training_config.get('early_stopping', {}):
        early_stopping_callback = EarlyStopping(
            monitor=training_config['early_stopping'].get('monitor', 'val_map50'),
            patience=training_config['early_stopping'].get('patience', 10),
            mode=training_config['early_stopping'].get('mode', 'max'),
            min_delta=training_config['early_stopping'].get('min_delta', 0.001)
        )
        callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Configure trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=training_config.get('epochs', 50),
        min_epochs=training_config.get('min_epochs', 5),
        precision='16-mixed' if training_config.get('mixed_precision', True) else '32',
        gradient_clip_val=training_config.get('gradient_clip_val', 10.0),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 1),
        check_val_every_n_epoch=training_config.get('check_val_every_n_epoch', 1),
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, datamodule=data_module)
    
    # Test model
    test_results = trainer.test(model, datamodule=data_module)[0]
    
    # Log test metrics
    for metric_name, metric_value in test_results.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Save best model path
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        mlflow.log_artifact(best_model_path, artifact_path="models")
        logging.info(f"Best model saved at: {best_model_path}")
    
    # Convert model to ONNX
    if mlflow_config.get('log_artifacts', {}).get('model', True):
        onnx_path = output_path / "model.onnx"
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, 640, 640)
            torch.onnx.export(
                model.model.model,
                dummy_input,
                onnx_path,
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
            mlflow.log_artifact(str(onnx_path), artifact_path="models")
            logging.info(f"ONNX model saved at: {onnx_path}")
        except Exception as e:
            logging.error(f"Failed to export ONNX model: {e}")
    
    # Gather results
    training_results = {
        'best_model_path': best_model_path if best_model_path else None,
        'best_score': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        'test_results': test_results,
        'mlflow_run_id': logger.run_id
    }
    
    return model, training_results


def setup_mlflow(mlflow_config: Dict[str, Any]) -> None:
    """Set up MLflow tracking."""
    tracking_uri = mlflow_config.get('tracking_uri', "http://127.0.0.1:8080")
    experiment_name = mlflow_config.get('experiment_name', "barcode_detection")
    
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logging.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            logging.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
    except Exception as e:
        logging.warning(f"Failed to set up MLflow experiment: {e}")
        
    # Configure autologging
    if mlflow_config.get('autolog', {}).get('pytorch_lightning', True):
        mlflow.pytorch.autolog(
            log_models=mlflow_config.get('autolog', {}).get('log_models', True),
            log_every_n_epoch=mlflow_config.get('autolog', {}).get('log_every_n_epoch', 1)
        )


def log_git_info() -> None:
    """Log git repository information to MLflow."""
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_id = repo.head.object.hexsha
        branch_name = repo.active_branch.name
        repo_url = next((remote.url for remote in repo.remotes), None)
        
        mlflow.set_tag("git.commit", commit_id)
        mlflow.set_tag("git.branch", branch_name)
        if repo_url:
            mlflow.set_tag("git.repo", repo_url)
            
        logging.info(f"Logged git info: commit={commit_id}, branch={branch_name}")
    except Exception as e:
        logging.warning(f"Failed to log git info: {e}")


def save_model_artifacts(
    model: pl.LightningModule,
    output_dir: str,
    include_onnx: bool = True,
    include_torchscript: bool = False
) -> Dict[str, str]:
    """
    Save model artifacts for production.
    
    Args:
        model: Trained PyTorch Lightning model
        output_dir: Directory to save artifacts
        include_onnx: Whether to save ONNX format
        include_torchscript: Whether to save TorchScript format
        
    Returns:
        Dictionary of saved artifact paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    artifacts = {}
    
    # Save PyTorch model
    torch_path = output_path / "model.pt"
    try:
        torch.save(model.state_dict(), torch_path)
        artifacts['pytorch'] = str(torch_path)
        logging.info(f"PyTorch model saved to {torch_path}")
    except Exception as e:
        logging.error(f"Failed to save PyTorch model: {e}")
    
    # Save ONNX model
    if include_onnx:
        onnx_path = output_path / "model.onnx"
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, 640, 640)
            torch.onnx.export(
                model.model.model,
                dummy_input,
                onnx_path,
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
            artifacts['onnx'] = str(onnx_path)
            logging.info(f"ONNX model saved to {onnx_path}")
        except Exception as e:
            logging.error(f"Failed to export ONNX model: {e}")
    
    # Save TorchScript model
    if include_torchscript:
        script_path = output_path / "model.pt"
        try:
            model.eval()
            scripted_model = torch.jit.script(model)
            scripted_model.save(script_path)
            artifacts['torchscript'] = str(script_path)
            logging.info(f"TorchScript model saved to {script_path}")
        except Exception as e:
            logging.error(f"Failed to export TorchScript model: {e}")
    
    return artifacts
