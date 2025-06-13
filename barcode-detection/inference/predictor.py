"""Predictor class for barcode detection inference"""

import json
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

from ultralytics import YOLO
import onnxruntime as ort

from barcode_detection.models.metrics import calculate_metrics
from barcode_detection.data.transforms import BasicTransforms


class BarcodePredictor:
    """Predictor class for barcode detection."""
    
    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model file (.pt, .onnx, or .trt)
            config: Inference configuration
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.config = config
        self.device = device
        
        # Model settings
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.max_detections = config.get('max_detections', 100)
        self.input_size = config.get('input_size', 640)
        
        # Class names
        self.class_names = ['qr', 'datamatrix', 'pdf417', 'ean13', 'other']
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self) -> Union[YOLO, ort.InferenceSession]:
        """Load model based on file extension."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        model_type = self.model_path.suffix.lower()
        
        if model_type == '.pt':
            # PyTorch model
            model = YOLO(str(self.model_path))
            if self.device != 'cpu':
                model.to(self.device)
            return model
            
        elif model_type == '.onnx':
            # ONNX model
            providers = ['CPUExecutionProvider']
            if self.device.startswith('cuda') and ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
                
            return ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            
        elif model_type == '.trt':
            # TensorRT model (would need additional implementation)
            raise NotImplementedError("TensorRT inference not implemented yet")
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def predict_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Predict barcodes in a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape[:2]
        
        # Run inference
        detections = self._run_inference(image_rgb)
        
        # Post-process results
        results = {
            'image_path': str(image_path),
            'image_shape': original_shape,
            'detections': detections,
            'processing_time': time.time() - start_time
        }
        
        return results
    
    def predict_batch(self, input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Predict barcodes in multiple images.
        
        Args:
            input_dir: Directory containing input images
            
        Returns:
            List of detection results for each image
        """
        input_path = Path(input_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        results = []
        for image_file in sorted(image_files):
            try:
                result = self.predict_image(image_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                
        return results
    
    def _run_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run model inference on preprocessed image."""
        if isinstance(self.model, YOLO):
            return self._run_pytorch_inference(image)
        elif isinstance(self.model, ort.InferenceSession):
            return self._run_onnx_inference(image)
        else:
            raise ValueError("Unknown model type")
    
    def _run_pytorch_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference with PyTorch YOLO model."""
        # Run YOLO inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i].tolist(),
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else 'unknown',
                        'corners': self._bbox_to_corners(boxes[i])
                    }
                    detections.append(detection)
                    
        return detections
    
    def _run_onnx_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference with ONNX model."""
        # Preprocess image for ONNX
        input_tensor = self._preprocess_for_onnx(image)
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})
        
        # Post-process ONNX outputs
        return self._postprocess_onnx_outputs(outputs[0], image.shape[:2])
    
    def _preprocess_for_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model."""
        # Resize image
        resized = BasicTransforms.resize_with_padding(image, self.input_size)
        
        # Normalize
        normalized = BasicTransforms.normalize(resized)
        
        # Convert to CHW format and add batch dimension
        tensor = BasicTransforms.to_tensor(normalized)
        batch_tensor = tensor.unsqueeze(0).numpy()
        
        return batch_tensor
    
    def _postprocess_onnx_outputs(self, outputs: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Post-process ONNX model outputs."""
        detections = []
        
        # ONNX output format depends on the specific YOLO variant
        # This is a generic implementation - may need adjustment
        for detection in outputs:
            # Extract box coordinates, confidence, and class
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, cls = detection[:6]
                
                if conf >= self.confidence_threshold:
                    # Convert coordinates back to original image scale
                    h, w = original_shape
                    scale_x = w / self.input_size
                    scale_y = h / self.input_size
                    
                    bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                    
                    detection_dict = {
                        'bbox': bbox,
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else 'unknown',
                        'corners': self._bbox_to_corners(bbox)
                    }
                    detections.append(detection_dict)
                    
        return detections
    
    def _bbox_to_corners(self, bbox: Union[np.ndarray, List[float]]) -> List[List[float]]:
        """Convert bounding box to corner coordinates."""
        x1, y1, x2, y2 = bbox
        
        # Return corners in clockwise order starting from bottom-left
        corners = [
            [x1, y2],  # bottom-left
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2]   # bottom-right
        ]
        
        return corners
    
    def evaluate(self, test_data_path: Union[str, Path]) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_data_path: Path to test data directory
            
        Returns:
            Dictionary of evaluation metrics
        """
        test_path = Path(test_data_path)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_path.glob(f"*{ext}"))
            
        all_predictions = []
        all_ground_truths = []
        
        for image_file in image_files:
            # Get predictions
            result = self.predict_image(image_file)
            pred_boxes = [det['bbox'] for det in result['detections']]
            pred_confidences = [det['confidence'] for det in result['detections']]
            pred_labels = [det['class_id'] for det in result['detections']]
            
            all_predictions.append({
                'boxes': np.array(pred_boxes) if pred_boxes else np.array([]).reshape(0, 4),
                'scores': np.array(pred_confidences) if pred_confidences else np.array([]),
                'labels': np.array(pred_labels) if pred_labels else np.array([])
            })
            
            # Load ground truth
            annotation_file = image_file.with_suffix('.json')
            if annotation_file.exists():
                gt_boxes, gt_labels = self._load_ground_truth(annotation_file)
                all_ground_truths.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })
            else:
                all_ground_truths.append({
                    'boxes': np.array([]).reshape(0, 4),
                    'labels': np.array([])
                })
        
        # Calculate metrics
        pred_boxes = [pred['boxes'] for pred in all_predictions]
        pred_scores = [pred['scores'] for pred in all_predictions]
        pred_labels = [pred['labels'] for pred in all_predictions]
        true_boxes = [gt['boxes'] for gt in all_ground_truths]
        true_labels = [gt['labels'] for gt in all_ground_truths]
        
        metrics = calculate_metrics(
            pred_boxes, pred_scores, pred_labels,
            true_boxes, true_labels
        )
        
        return metrics
    
    def _load_ground_truth(self, annotation_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load ground truth annotations from JSON file."""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        boxes = []
        labels = []
        
        for obj in data.get('objects', []):
            coords = obj.get('data', [])
            
            if len(coords) == 4:
                # Convert polygon to bounding box
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)
                
                boxes.append([x_min, y_min, x_max, y_max])
                
                # Map class name to ID
                class_name = obj.get('class', 'other')
                class_mapping = {name: i for i, name in enumerate(self.class_names)}
                class_id = class_mapping.get(class_name, len(self.class_names) - 1)
                labels.append(class_id)
        
        return (
            np.array(boxes) if boxes else np.array([]).reshape(0, 4),
            np.array(labels) if labels else np.array([])
        )
    
    def save_results(self, results: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save detection results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def save_batch_results(self, results: List[Dict[str, Any]], output_dir: Union[str, Path]) -> None:
        """Save batch detection results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for result in results:
            image_name = Path(result['image_path']).stem
            output_file = output_path / f"{image_name}_results.json"
            self.save_results(result, output_file)
        
        # Save summary
        summary = {
            'total_images': len(results),
            'total_detections': sum(len(r['detections']) for r in results),
            'average_processing_time': np.mean([r['processing_time'] for r in results]),
            'results': results
        }
        
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def visualize_detections(self, image_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """Visualize detections on image and save."""
        # Get predictions
        results = self.predict_image(image_path)
        
        # Load image
        image = cv2.imread(str(image_path))
        
        # Draw detections
        for detection in results['detections']:
            bbox = detection['bbox']
            corners = detection['corners']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )
            
            # Draw corners
            corners_np = np.array(corners, dtype=np.int32)
            cv2.polylines(image, [corners_np], True, (255, 0, 0), 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                image,
                label,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Save result
        cv2.imwrite(str(output_path), image)
