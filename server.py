"""FastAPI server for barcode detection inference"""

import io
import json
import logging
import uvicorn
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from barcode_detection.inference.predictor import BarcodePredictor


class InferenceServer:
    """FastAPI server for barcode detection inference"""
    
    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Initialize inference server.
        
        Args:
            model_path: Path to model file
            config: Inference configuration
            device: Device to run inference on
        """
        self.predictor = BarcodePredictor(model_path, config, device)
        self.app = FastAPI(
            title="Barcode Detection API",
            description="API for detecting barcodes and QR codes in images",
            version="1.0.0"
        )
        
        # Set up routes
        self._setup_routes()
        
    def _setup_routes(self) -> None:
        """Set up API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {"message": "Barcode Detection API", "status": "running"}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "model_loaded": True}
        
        @self.app.get("/info")
        async def model_info():
            """Get model information."""
            return {
                "model_path": str(self.predictor.model_path),
                "device": self.predictor.device,
                "class_names": self.predictor.class_names,
                "confidence_threshold": self.predictor.confidence_threshold,
                "iou_threshold": self.predictor.iou_threshold,
                "input_size": self.predictor.input_size
            }
        
        @self.app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            """
            Predict barcodes in uploaded image.
            
            Args:
                file: Uploaded image file
                
            Returns:
                JSON response with detection results
            """
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail="File must be an image"
                    )
                
                # Read image
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to numpy array
                image_array = np.array(image)
                
                # Run inference
                results = self.predictor._run_inference(image_array)
                
                # Prepare response
                response = {
                    "filename": file.filename,
                    "image_shape": image_array.shape[:2],
                    "detections": results,
                    "total_detections": len(results)
                }
                
                return JSONResponse(content=response)
                
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction failed: {str(e)}"
                )
        
        @self.app.post("/predict_batch")
        async def predict_batch(files: List[UploadFile] = File(...)):
            """
            Predict barcodes in multiple uploaded images.
            
            Args:
                files: List of uploaded image files
                
            Returns:
                JSON response with batch detection results
            """
            try:
                results = []
                
                for file in files:
                    # Validate file type
                    if not file.content_type.startswith('image/'):
                        continue
                    
                    # Process each image
                    image_bytes = await file.read()
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image_array = np.array(image)
                    
                    # Run inference
                    detections = self.predictor._run_inference(image_array)
                    
                    # Add to results
                    result = {
                        "filename": file.filename,
                        "image_shape": image_array.shape[:2],
                        "detections": detections,
                        "total_detections": len(detections)
                    }
                    results.append(result)
                
                # Prepare batch response
                response = {
                    "total_images": len(results),
                    "total_detections": sum(r["total_detections"] for r in results),
                    "results": results
                }
                
                return JSONResponse(content=response)
                
            except Exception as e:
                logging.error(f"Batch prediction error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch prediction failed: {str(e)}"
                )
        
        @self.app.post("/predict_url")
        async def predict_from_url(image_url: str):
            """
            Predict barcodes from image URL.
            
            Args:
                image_url: URL of the image to process
                
            Returns:
                JSON response with detection results
            """
            try:
                import requests
                from PIL import Image
                
                # Download image
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Open image
                image = Image.open(io.BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_array = np.array(image)
                
                # Run inference
                results = self.predictor._run_inference(image_array)
                
                # Prepare response
                response = {
                    "image_url": image_url,
                    "image_shape": image_array.shape[:2],
                    "detections": results,
                    "total_detections": len(results)
                }
                
                return JSONResponse(content=response)
                
            except Exception as e:
                logging.error(f"URL prediction error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"URL prediction failed: {str(e)}"
                )


def start_server(
    model_path: str,
    config: Dict[str, Any],
    host: str = "0.0.0.0",
    port: int = 8080,
    workers: int = 1,
    reload: bool = False
) -> None:
    """
    Start the inference server.
    
    Args:
        model_path: Path to model file
        config: Inference configuration
        host: Server host
        port: Server port
        workers: Number of worker processes
        reload: Enable auto-reload for development
    """
    # Create server instance
    server = InferenceServer(model_path, config)
    
    # Start server
    uvicorn.run(
        server.app,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )


def create_app(model_path: str, config: Dict[str, Any]) -> FastAPI:
    """
    Create FastAPI app for deployment.
    
    Args:
        model_path: Path to model file
        config: Inference configuration
        
    Returns:
        FastAPI application instance
    """
    server = InferenceServer(model_path, config)
    return server.app


# For use with ASGI servers like Gunicorn
def create_production_app():
    """Create app for production deployment."""
    import os
    from omegaconf import OmegaConf
    
    # Load configuration
    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    cfg = OmegaConf.load(config_path)
    
    # Get model path
    model_path = os.getenv("MODEL_PATH", cfg.inference.model_path)
    
    # Create app
    return create_app(
        model_path=model_path,
        config=OmegaConf.to_container(cfg.inference, resolve=True)
    )


if __name__ == "__main__":
    # For development/testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python server.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Default config for standalone run
    config = {
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'max_detections': 100,
        'input_size': 640
    }
    
    start_server(model_path, config, port=8080)
