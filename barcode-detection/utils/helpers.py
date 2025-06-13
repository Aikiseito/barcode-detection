"""Helper utilities for barcode detection project"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import dvc.api
import logging

logger = logging.getLogger(__name__)


def ensure_directories(directories: List[str]) -> None:
    """
    Ensure that directories exist, create if they don't.
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")


def download_file(url: str, output_path: Union[str, Path]) -> None:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download
        output_path: Path to save downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {url} to {output_path}")
    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Download completed: {output_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def download_and_extract_zip(url: str, extract_dir: Union[str, Path]) -> None:
    """
    Download and extract a zip file.
    
    Args:
        url: URL to download
        extract_dir: Directory to extract files to
    """
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Download to temporary file
    tmp_file = extract_dir / "temp.zip"
    download_file(url, tmp_file)
    
    # Extract
    logger.info(f"Extracting {tmp_file} to {extract_dir}")
    try:
        with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Extraction completed: {extract_dir}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
    finally:
        # Clean up
        if tmp_file.exists():
            tmp_file.unlink()


def get_dvc_data_info() -> Dict[str, Any]:
    """
    Get information about DVC-tracked data.
    
    Returns:
        Dictionary with data information
    """
    info = {
        'train': None,
        'val': None,
        'test': None
    }
    
    try:
        # Check for train data
        try:
            train_info = dvc.api.get_url('data/train.dvc')
            info['train'] = {'url': train_info}
        except Exception:
            logger.debug("No DVC info found for train data")
        
        # Check for val data
        try:
            val_info = dvc.api.get_url('data/val.dvc')
            info['val'] = {'url': val_info}
        except Exception:
            logger.debug("No DVC info found for val data")
        
        # Check for test data
        try:
            test_info = dvc.api.get_url('data/test.dvc')
            info['test'] = {'url': test_info}
        except Exception:
            logger.debug("No DVC info found for test data")
            
    except Exception as e:
        logger.warning(f"Error getting DVC data info: {e}")
    
    return info


def download_data(
    output_dir: Union[str, Path] = "data",
    force: bool = False
) -> None:
    """
    Download sample data for testing if needed.
    This function should be adapted for your specific dataset.
    
    Args:
        output_dir: Directory to save data
        force: Force download even if data already exists
    """
    # Since the user mentioned their dataset is on local machine and cannot be downloaded,
    # this function is a placeholder. In a real scenario, you would implement
    # logic to download data from a public source.
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    train_path = output_path / "train"
    val_path = output_path / "val"
    test_path = output_path / "test"
    
    if (train_path.exists() and val_path.exists() and test_path.exists()) and not force:
        logger.info("Data directories already exist, skipping download")
        return
    
    logger.info("This is a placeholder for data download.")
    logger.info("Since your dataset is local, please manually place it in the correct directories:")
    logger.info(f"- Training data: {train_path}")
    logger.info(f"- Validation data: {val_path}")
    logger.info(f"- Test data: {test_path}")
    logger.info("Then initialize DVC tracking with:")
    logger.info("dvc add data/train data/val data/test")
    
    # Create empty directories
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)


def convert_json_to_yolo(
    json_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    class_mapping: Optional[Dict[str, int]] = None
) -> None:
    """
    Convert JSON annotations to YOLO format.
    
    Args:
        json_dir: Directory with JSON files
        output_dir: Output directory for YOLO annotations (default: same as json_dir)
        class_mapping: Class name to ID mapping
    """
    import json
    
    json_dir = Path(json_dir)
    if output_dir is None:
        output_dir = json_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default class mapping for barcode types
    if class_mapping is None:
        class_mapping = {
            'qr': 0,
            'datamatrix': 1,
            'pdf417': 2,
            'ean13': 3,
            'other': 4
        }
    
    # Process each JSON file
    json_files = list(json_dir.glob("*.json"))
    logger.info(f"Converting {len(json_files)} JSON files to YOLO format")
    
    for json_file in json_files:
        try:
            # Load JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Output file path
            output_file = output_dir / f"{json_file.stem}.txt"
            
            # Check if we have corresponding image
            image_found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                image_path = json_file.with_suffix(ext)
                if image_path.exists():
                    image_found = True
                    break
            
            if not image_found:
                logger.warning(f"No image found for {json_file}")
                continue
            
            # Process objects
            yolo_annotations = []
            for obj in data.get('objects', []):
                coords = obj.get('data', [])
                class_name = obj.get('class', 'other')
                
                # Get class ID
                class_id = class_mapping.get(class_name.lower(), class_mapping['other'])
                
                if len(coords) == 4:
                    # Convert polygon to bounding box
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                    
                    # Get image dimensions
                    import cv2
                    img = cv2.imread(str(image_path))
                    img_height, img_width = img.shape[:2]
                    
                    # Calculate YOLO format (normalized)
                    x_center = (x_min + x_max) / (2 * img_width)
                    y_center = (y_min + y_max) / (2 * img_height)
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    # Add to annotations
                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            # Write YOLO format file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_annotations))
                
            logger.debug(f"Converted {json_file} to {output_file}")
            
        except Exception as e:
            logger.error(f"Error converting {json_file}: {e}")
    
    logger.info(f"Conversion completed. YOLO annotations saved to {output_dir}")


def create_custom_data_yaml(
    data_dir: Union[str, Path],
    output_file: Union[str, Path] = "custom_data.yaml",
    class_names: Optional[List[str]] = None
) -> None:
    """
    Create custom_data.yaml for YOLO training.
    
    Args:
        data_dir: Data directory
        output_file: Output YAML file
        class_names: List of class names
    """
    import yaml
    
    data_dir = Path(data_dir)
    output_file = Path(output_file)
    
    # Default class names
    if class_names is None:
        class_names = ['qr', 'datamatrix', 'pdf417', 'ean13', 'other']
    
    # Create YAML content
    data_yaml = {
        'train': str(data_dir / 'train'),
        'val': str(data_dir / 'val'),
        'test': str(data_dir / 'test'),
        'nc': len(class_names),
        'names': class_names
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    logger.info(f"Created YOLO data configuration: {output_file}")


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Set random seed: {seed}")


def get_git_hash() -> str:
    """
    Get current git commit hash.
    
    Returns:
        Git commit hash or 'unknown'
    """
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "unknown"
