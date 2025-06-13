"""Dataset class for barcode detection"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image


class BarcodeDataset(Dataset):
    """Dataset class for barcode and QR code detection."""
    
    def __init__(
        self,
        data_dir: Path,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
        annotation_format: str = 'json',
        transforms: Optional[Callable] = None,
        image_size: int = 640,
        class_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images and annotations
            image_extensions: List of valid image file extensions
            annotation_format: Format of annotations ('json' or 'yolo')
            transforms: Transform functions to apply
            image_size: Target image size
            class_mapping: Mapping from class names to indices
        """
        self.data_dir = Path(data_dir)
        self.image_extensions = image_extensions
        self.annotation_format = annotation_format
        self.transforms = transforms
        self.image_size = image_size
        
        # Default class mapping for barcode types
        if class_mapping is None:
            self.class_mapping = {
                'qr': 0,
                'datamatrix': 1, 
                'pdf417': 2,
                'ean13': 3,
                'other': 4
            }
        else:
            self.class_mapping = class_mapping
            
        self.class_names = list(self.class_mapping.keys())
        self.num_classes = len(self.class_names)
        
        # Load image paths and annotations
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all samples (image paths and corresponding annotations)"""
        samples = []
        
        # Find all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.data_dir.glob(f"*{ext}"))
            
        for image_path in sorted(image_files):
            # Find corresponding annotation file
            annotation_path = self._get_annotation_path(image_path)
            
            if annotation_path and annotation_path.exists():
                samples.append({
                    'image_path': image_path,
                    'annotation_path': annotation_path
                })
            else:
                print(f"Warning: No annotation found for {image_path}")
                
        print(f"Loaded {len(samples)} samples from {self.data_dir}")
        return samples
        
    def _get_annotation_path(self, image_path: Path) -> Optional[Path]:
        """Get the annotation file path for a given image"""
        base_name = image_path.stem
        
        if self.annotation_format == 'json':
            return image_path.parent / f"{base_name}.json"
        elif self.annotation_format == 'yolo':
            return image_path.parent / f"{base_name}.txt"
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
            
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Load annotations
        target = self._load_annotations(sample['annotation_path'], image.shape[:2])
        
        # Apply transforms
        if self.transforms:
            image, target = self.transforms(image, target)
            
        # Convert image to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        return image, target
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess an image."""
        # Load image using OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image while maintaining aspect ratio
        image = self._resize_image(image, self.image_size)
        
        return image
        
    def _resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_size / w, target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
        
    def _load_annotations(self, annotation_path: Path, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Load annotations from file."""
        if self.annotation_format == 'json':
            return self._load_json_annotations(annotation_path, image_shape)
        elif self.annotation_format == 'yolo':
            return self._load_yolo_annotations(annotation_path, image_shape)
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
            
    def _load_json_annotations(self, annotation_path: Path, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Load annotations from JSON file."""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        boxes = []
        labels = []
        
        # Extract bounding boxes from objects
        for obj in data.get('objects', []):
            coords = obj.get('data', [])
            
            if len(coords) == 4:
                # Convert polygon coordinates to bounding box
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)
                
                # Normalize coordinates to [0, 1]
                h, w = image_shape
                x_min_norm = x_min / w
                y_min_norm = y_min / h
                x_max_norm = x_max / w
                y_max_norm = y_max / h
                
                boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
                
                # Get class label (assuming it's in the object data)
                class_name = obj.get('class', 'other')
                class_id = self.class_mapping.get(class_name, self.class_mapping['other'])
                labels.append(class_id)
                
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([hash(str(annotation_path))], dtype=torch.long),
            'area': self._calculate_area(boxes),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.uint8)
        }
        
        return target
        
    def _load_yolo_annotations(self, annotation_path: Path, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Load annotations from YOLO format file."""
        boxes = []
        labels = []
        
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from YOLO format (center, width, height) to xyxy
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)
                
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([hash(str(annotation_path))], dtype=torch.long),
            'area': self._calculate_area(boxes),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.uint8)
        }
        
        return target
        
    def _calculate_area(self, boxes: torch.Tensor) -> torch.Tensor:
        """Calculate area of bounding boxes."""
        if len(boxes) == 0:
            return torch.zeros((0,), dtype=torch.float32)
            
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return area
        
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        for name, id_ in self.class_mapping.items():
            if id_ == class_id:
                return name
        return 'unknown'
        
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample."""
        sample = self.samples[idx]
        return {
            'image_path': str(sample['image_path']),
            'annotation_path': str(sample['annotation_path']),
            'index': idx
        }
