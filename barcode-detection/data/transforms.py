"""Data transforms for barcode detection"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Callable, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(data_config: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Get training and validation transforms.
    
    Args:
        data_config: Configuration dictionary containing augmentation settings
        
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    image_size = data_config.get('image_size', 640)
    augmentation_config = data_config.get('augmentation', {})
    
    # Training transforms with augmentation
    train_transforms = []
    
    # Always resize to target size
    train_transforms.append(
        A.Resize(height=image_size, width=image_size, p=1.0)
    )
    
    # Add augmentations if enabled
    if augmentation_config.get('enabled', True):
        # Horizontal flip
        if augmentation_config.get('horizontal_flip', 0) > 0:
            train_transforms.append(
                A.HorizontalFlip(p=augmentation_config['horizontal_flip'])
            )
            
        # Vertical flip
        if augmentation_config.get('vertical_flip', 0) > 0:
            train_transforms.append(
                A.VerticalFlip(p=augmentation_config['vertical_flip'])
            )
            
        # Rotation
        if augmentation_config.get('rotation', 0) > 0:
            train_transforms.append(
                A.Rotate(
                    limit=augmentation_config['rotation'],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                )
            )
            
        # Color augmentations
        color_augs = []
        if augmentation_config.get('brightness', 0) > 0:
            color_augs.append(
                A.RandomBrightness(
                    limit=augmentation_config['brightness'],
                    p=0.5
                )
            )
            
        if augmentation_config.get('contrast', 0) > 0:
            color_augs.append(
                A.RandomContrast(
                    limit=augmentation_config['contrast'],
                    p=0.5
                )
            )
            
        if color_augs:
            train_transforms.extend(color_augs)
            
        # Additional augmentations
        train_transforms.extend([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            )
        ])
    
    # Normalization and tensor conversion
    train_transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = [
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ]
    
    # Create composition
    train_transform = A.Compose(
        train_transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        )
    )
    
    val_transform = A.Compose(
        val_transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        )
    )
    
    # Wrap transforms to handle our data format
    def wrapped_train_transform(image: np.ndarray, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return apply_albumentations_transform(train_transform, image, target)
        
    def wrapped_val_transform(image: np.ndarray, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return apply_albumentations_transform(val_transform, image, target)
    
    return wrapped_train_transform, wrapped_val_transform


def apply_albumentations_transform(
    transform: A.Compose,
    image: np.ndarray,
    target: Dict[str, Any]
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply Albumentations transform to image and target.
    
    Args:
        transform: Albumentations transform composition
        image: Input image as numpy array
        target: Target dictionary containing boxes and labels
        
    Returns:
        Tuple of transformed (image, target)
    """
    # Extract boxes and labels
    boxes = target.get('boxes', torch.tensor([]))
    labels = target.get('labels', torch.tensor([]))
    
    # Convert boxes to format expected by Albumentations (x_min, y_min, x_max, y_max)
    if len(boxes) > 0:
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.numpy()
        else:
            boxes_np = np.array(boxes)
            
        # Ensure boxes are in correct format
        if boxes_np.shape[1] == 4:
            # Convert normalized coordinates to pixel coordinates
            h, w = image.shape[:2]
            boxes_pixel = boxes_np.copy()
            boxes_pixel[:, 0] *= w  # x_min
            boxes_pixel[:, 1] *= h  # y_min
            boxes_pixel[:, 2] *= w  # x_max
            boxes_pixel[:, 3] *= h  # y_max
        else:
            boxes_pixel = boxes_np
            
        # Convert labels to list
        if isinstance(labels, torch.Tensor):
            labels_list = labels.tolist()
        else:
            labels_list = list(labels)
    else:
        boxes_pixel = []
        labels_list = []
    
    # Apply transform
    try:
        transformed = transform(
            image=image,
            bboxes=boxes_pixel,
            class_labels=labels_list
        )
        
        transformed_image = transformed['image']
        transformed_boxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']
        
        # Convert back to our format
        if transformed_boxes:
            # Convert back to normalized coordinates
            if isinstance(transformed_image, torch.Tensor):
                h, w = transformed_image.shape[1], transformed_image.shape[2]
            else:
                h, w = transformed_image.shape[:2]
                
            boxes_norm = np.array(transformed_boxes)
            boxes_norm[:, 0] /= w  # x_min
            boxes_norm[:, 1] /= h  # y_min
            boxes_norm[:, 2] /= w  # x_max
            boxes_norm[:, 3] /= h  # y_max
            
            new_boxes = torch.tensor(boxes_norm, dtype=torch.float32)
            new_labels = torch.tensor(transformed_labels, dtype=torch.long)
        else:
            new_boxes = torch.zeros((0, 4), dtype=torch.float32)
            new_labels = torch.zeros((0,), dtype=torch.long)
            
    except Exception as e:
        print(f"Transform failed: {e}")
        # Fallback to original data
        if isinstance(image, np.ndarray):
            transformed_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            transformed_image = image
        new_boxes = boxes
        new_labels = labels
    
    # Update target
    new_target = target.copy()
    new_target['boxes'] = new_boxes
    new_target['labels'] = new_labels
    
    # Recalculate area if needed
    if len(new_boxes) > 0:
        new_target['area'] = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
    else:
        new_target['area'] = torch.zeros((0,), dtype=torch.float32)
    
    return transformed_image, new_target


class BasicTransforms:
    """Basic transforms without Albumentations dependency."""
    
    @staticmethod
    def resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image with padding to maintain aspect ratio."""
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
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def to_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        if len(image.shape) == 3:
            # HWC to CHW
            return torch.from_numpy(image).permute(2, 0, 1)
        else:
            return torch.from_numpy(image)


def get_basic_transforms(image_size: int = 640) -> Tuple[Callable, Callable]:
    """
    Get basic transforms without Albumentations dependency.
    
    Args:
        image_size: Target image size
        
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    def train_transform(image: np.ndarray, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Basic preprocessing
        image = BasicTransforms.resize_with_padding(image, image_size)
        image = BasicTransforms.normalize(image)
        image = BasicTransforms.to_tensor(image)
        
        return image, target
    
    def val_transform(image: np.ndarray, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Same as training but without augmentation
        image = BasicTransforms.resize_with_padding(image, image_size)
        image = BasicTransforms.normalize(image)
        image = BasicTransforms.to_tensor(image)
        
        return image, target
    
    return train_transform, val_transform
