"""Conversion utilities for producing production-ready models."""

import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import onnx
import numpy as np
from omegaconf import OmegaConf

from barcode_detection.models.yolo_lightning import YOLOLightning
from barcode_detection.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def convert_to_onnx(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    input_shape: tuple = (1, 3, 640, 640),
    dynamic_axes: bool = True,
    opset_version: int = 11,
    simplify: bool = True
) -> None:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model
        output_path: Output path for ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        dynamic_axes: Enable dynamic axes for variable batch size
        opset_version: ONNX opset version
        simplify: Apply ONNX simplification
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting {model_path} to ONNX format")
    
    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    try:
        if model_path.suffix == '.pt':
            if 'yolov5' in str(model_path) or 'yolov8' in str(model_path):
                # Handle Ultralytics models
                from ultralytics import YOLO
                model = YOLO(str(model_path))
                
                # Export directly using YOLO's export method
                model.export(format='onnx', opset=opset_version, simplify=simplify)
                
                # Move the generated file to the desired output path
                generated_onnx = model_path.with_suffix('.onnx')
                if generated_onnx.exists() and generated_onnx != output_path:
                    import shutil
                    shutil.move(str(generated_onnx), str(output_path))
                
                logger.info(f"ONNX model saved to {output_path}")
                return
            else:
                # Handle PyTorch Lightning models
                model = YOLOLightning.load_from_checkpoint(str(model_path))
                model.eval()
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Define dynamic axes if enabled
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model.model.model,  # Access the YOLO model within the Lightning wrapper
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict
        )
        
        logger.info(f"PyTorch model exported to ONNX: {output_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verified successfully")
        
        # Simplify ONNX model if requested
        if simplify:
            try:
                import onnxsim
                
                logger.info("Simplifying ONNX model...")
                simplified_model, check = onnxsim.simplify(onnx_model)
                
                if check:
                    onnx.save(simplified_model, output_path)
                    logger.info(f"Simplified ONNX model saved to {output_path}")
                else:
                    logger.warning("Failed to validate simplified ONNX model, using original model")
            except ImportError:
                logger.warning("onnxsim not installed, skipping ONNX simplification")
            except Exception as e:
                logger.warning(f"ONNX simplification failed: {e}")
        
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise


def convert_onnx_to_tensorrt(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    precision: str = "fp16",
    workspace_size: int = 1,
    max_batch_size: int = 1,
    min_shape: Optional[tuple] = None,
    opt_shape: Optional[tuple] = None,
    max_shape: Optional[tuple] = None
) -> None:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output path for TensorRT engine
        precision: Precision mode (fp32, fp16, int8)
        workspace_size: Workspace size in GB
        max_batch_size: Maximum batch size
        min_shape: Minimum input shape (excluding batch dimension)
        opt_shape: Optimal input shape (excluding batch dimension)
        max_shape: Maximum input shape (excluding batch dimension)
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT is not installed. Please install TensorRT to use this feature.")
        return
    
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting {onnx_path} to TensorRT engine")
    
    # Check if ONNX model exists
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    # Set default shapes if not provided
    if min_shape is None:
        min_shape = (3, 640, 640)
    if opt_shape is None:
        opt_shape = (3, 640, 640)
    if max_shape is None:
        max_shape = (3, 640, 640)
    
    # Create TensorRT logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parser error: {parser.get_error(error)}")
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size * (1 << 30)  # Convert GB to bytes
    
    # Set precision
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Using FP16 precision")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("Using INT8 precision")
    else:
        logger.info("Using FP32 precision")
    
    # Set optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        (1, *min_shape),
        (max_batch_size // 2, *opt_shape),
        (max_batch_size, *max_shape)
    )
    config.add_optimization_profile(profile)
    
    # Build engine
    logger.info("Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    logger.info(f"TensorRT engine saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Convert models to production formats")
    
    # Create subparsers for different conversion types
    subparsers = parser.add_subparsers(dest="command", help="Conversion command")
    
    # ONNX conversion
    onnx_parser = subparsers.add_parser("onnx", help="Convert to ONNX format")
    onnx_parser.add_argument("--model_path", required=True, help="Path to PyTorch model")
    onnx_parser.add_argument("--output_path", help="Output path for ONNX model")
    onnx_parser.add_argument("--input_size", type=int, default=640, help="Input size (default: 640)")
    onnx_parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    onnx_parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes")
    onnx_parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11)")
    onnx_parser.add_argument("--no_simplify", action="store_true", help="Disable ONNX simplification")
    
    # TensorRT conversion
    trt_parser = subparsers.add_parser("tensorrt", help="Convert to TensorRT format")
    trt_parser.add_argument("--onnx_path", required=True, help="Path to ONNX model")
    trt_parser.add_argument("--output_path", help="Output path for TensorRT engine")
    trt_parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16", help="Precision mode")
    trt_parser.add_argument("--workspace", type=int, default=1, help="Workspace size in GB (default: 1)")
    trt_parser.add_argument("--max_batch_size", type=int, default=1, help="Maximum batch size (default: 1)")
    trt_parser.add_argument("--input_size", type=int, default=640, help="Input size (default: 640)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level="INFO")
    
    if args.command == "onnx":
        # Default output path if not specified
        if args.output_path is None:
            model_path = Path(args.model_path)
            args.output_path = model_path.with_suffix('.onnx')
        
        # Convert to ONNX
        convert_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            input_shape=(args.batch_size, 3, args.input_size, args.input_size),
            dynamic_axes=args.dynamic,
            opset_version=args.opset,
            simplify=not args.no_simplify
        )
    
    elif args.command == "tensorrt":
        # Default output path if not specified
        if args.output_path is None:
            onnx_path = Path(args.onnx_path)
            args.output_path = onnx_path.with_suffix('.trt')
        
        # Convert to TensorRT
        convert_onnx_to_tensorrt(
            onnx_path=args.onnx_path,
            output_path=args.output_path,
            precision=args.precision,
            workspace_size=args.workspace,
            max_batch_size=args.max_batch_size,
            min_shape=(3, args.input_size, args.input_size),
            opt_shape=(3, args.input_size, args.input_size),
            max_shape=(3, args.input_size, args.input_size)
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
