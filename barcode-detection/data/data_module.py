"""PyTorch Lightning Data Module for barcode detection"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .dataset import BarcodeDataset
from .transforms import get_transforms


class BarcodeDataModule(pl.LightningDataModule):
    """PyTorch Lightning Data Module for barcode detection"""
    
    def __init__(
        self,
        data_config: Dict[str, Any],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        super().__init__()
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Data paths
        self.data_dir = Path(data_config.get('data_dir', 'data'))
        self.train_path = self.data_dir / data_config.get('train_path', 'train')
        self.val_path = self.data_dir / data_config.get('val_path', 'val')
        self.test_path = self.data_dir / data_config.get('test_path', 'test')
        
        # Data settings
        self.image_size = data_config.get('image_size', 640)
        self.image_extensions = data_config.get('image_extensions', ['.jpg', '.jpeg', '.png'])
        self.annotation_format = data_config.get('annotation_format', 'json')
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self) -> None:
        """Download or prepare data if needed."""
        # Check if DVC data needs to be pulled
        if hasattr(self, 'dvc_pull') and self.dvc_pull:
            self._pull_dvc_data()
            
        # Validate data directories
        for split_path, split_name in [
            (self.train_path, 'train'),
            (self.val_path, 'val'), 
            (self.test_path, 'test')
        ]:
            if not split_path.exists():
                raise FileNotFoundError(f"{split_name} data directory not found: {split_path}")
                
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing."""
        # Get transforms
        train_transforms, val_transforms = get_transforms(self.data_config)
        
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = BarcodeDataset(
                data_dir=self.train_path,
                image_extensions=self.image_extensions,
                annotation_format=self.annotation_format,
                transforms=train_transforms,
                image_size=self.image_size
            )
            
            # Validation dataset
            self.val_dataset = BarcodeDataset(
                data_dir=self.val_path,
                image_extensions=self.image_extensions,
                annotation_format=self.annotation_format,
                transforms=val_transforms,
                image_size=self.image_size
            )
            
        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset = BarcodeDataset(
                data_dir=self.test_path,
                image_extensions=self.image_extensions,
                annotation_format=self.annotation_format,
                transforms=val_transforms,
                image_size=self.image_size
            )
            
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def predict_dataloader(self) -> DataLoader:
        """Return prediction dataloader."""
        return self.test_dataloader()
        
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
        """Custom collate function for object detection."""
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
            
        # Stack images
        images = torch.stack(images, dim=0)
        
        return images, targets
        
    def _pull_dvc_data(self) -> None:
        """Pull data using DVC."""
        try:
            import dvc.api
            dvc.api.pull()
            print("DVC data pulled successfully")
        except ImportError:
            print("DVC not available, skipping data pull")
        except Exception as e:
            print(f"Error pulling DVC data: {e}")
            
    def get_class_names(self) -> List[str]:
        """Return class names."""
        # Default class names for barcode detection
        return ['qr', 'datamatrix', 'pdf417', 'ean13', 'other']
        
    def get_num_classes(self) -> int:
        """Return number of classes."""
        return len(self.get_class_names())
        
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        stats = {}
        
        if self.train_dataset:
            stats['train_size'] = len(self.train_dataset)
            
        if self.val_dataset:
            stats['val_size'] = len(self.val_dataset)
            
        if self.test_dataset:
            stats['test_size'] = len(self.test_dataset)
            
        stats['num_classes'] = self.get_num_classes()
        stats['class_names'] = self.get_class_names()
        stats['image_size'] = self.image_size
        
        return stats
