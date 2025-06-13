"""Metrics calculation for object detection"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Union
from scipy.spatial.distance import directed_hausdorff


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        box1: Bounding box in format [x1, y1, x2, y2]
        box2: Bounding box in format [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Check if there's no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area


def hausdorff_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Hausdorff distance between two bounding boxes.
    
    Args:
        bbox1: Bounding box in format [x1, y1, x2, y2]
        bbox2: Bounding box in format [x1, y1, x2, y2]
        
    Returns:
        Hausdorff distance
    """
    # Convert bounding boxes to corner points
    points1 = np.array([
        [bbox1[0], bbox1[1]],  # top-left
        [bbox1[2], bbox1[1]],  # top-right
        [bbox1[2], bbox1[3]],  # bottom-right
        [bbox1[0], bbox1[3]]   # bottom-left
    ])
    
    points2 = np.array([
        [bbox2[0], bbox2[1]],  # top-left
        [bbox2[2], bbox2[1]],  # top-right
        [bbox2[2], bbox2[3]],  # bottom-right
        [bbox2[0], bbox2[3]]   # bottom-left
    ])
    
    # Calculate directed Hausdorff distances
    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]
    
    # Return maximum (symmetric Hausdorff distance)
    return max(d1, d2)


def calculate_ap(
    predicted_boxes: List[np.ndarray],
    predicted_scores: List[np.ndarray], 
    predicted_labels: List[np.ndarray],
    true_boxes: List[np.ndarray],
    true_labels: List[np.ndarray],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision (AP) for a specific class
    
    Args:
        predicted_boxes: List of predicted bounding boxes for each image
        predicted_scores: List of prediction confidence scores for each image
        predicted_labels: List of predicted class labels for each image
        true_boxes: List of ground truth bounding boxes for each image
        true_labels: List of ground truth class labels for each image
        class_id: Class ID to calculate AP for
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Average Precision (AP) for the specified class
    """
    # Collect all predictions and ground truths for this class
    all_predictions = []
    all_ground_truths = []
    
    for i in range(len(predicted_boxes)):
        # Get predictions for this class
        if len(predicted_labels[i]) > 0:
            class_mask = predicted_labels[i] == class_id
            if np.any(class_mask):
                pred_boxes_class = predicted_boxes[i][class_mask]
                pred_scores_class = predicted_scores[i][class_mask]
                
                for j in range(len(pred_boxes_class)):
                    all_predictions.append({
                        'box': pred_boxes_class[j],
                        'score': pred_scores_class[j],
                        'image_id': i
                    })
        
        # Get ground truths for this class
        if len(true_labels[i]) > 0:
            gt_class_mask = true_labels[i] == class_id
            if np.any(gt_class_mask):
                gt_boxes_class = true_boxes[i][gt_class_mask]
                
                for j in range(len(gt_boxes_class)):
                    all_ground_truths.append({
                        'box': gt_boxes_class[j],
                        'image_id': i,
                        'matched': False
                    })
    
    if len(all_predictions) == 0:
        return 0.0
    
    # Sort predictions by confidence score (descending)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate precision and recall at each threshold
    true_positives = 0
    false_positives = 0
    num_ground_truths = len(all_ground_truths)
    
    precisions = []
    recalls = []
    
    for pred in all_predictions:
        # Find matching ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(all_ground_truths):
            if gt['image_id'] == pred['image_id'] and not gt['matched']:
                iou = calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        # Check if prediction is correct
        if best_iou >= iou_threshold:
            true_positives += 1
            all_ground_truths[best_gt_idx]['matched'] = True
        else:
            false_positives += 1
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / num_ground_truths if num_ground_truths > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    if len(precisions) == 0:
        return 0.0
    
    # Calculate AP using interpolation
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Add endpoints
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    
    # Interpolate precision values
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i-1] = max(precisions[i-1], precisions[i])
    
    # Calculate area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def calculate_map(
    predicted_boxes: List[np.ndarray],
    predicted_scores: List[np.ndarray],
    predicted_labels: List[np.ndarray], 
    true_boxes: List[np.ndarray],
    true_labels: List[np.ndarray],
    iou_threshold: float = 0.5,
    num_classes: int = 5
) -> float:
    """
    Calculate mean Average Precision (mAP) across all classes.
    
    Args:
        predicted_boxes: List of predicted bounding boxes for each image
        predicted_scores: List of prediction confidence scores for each image
        predicted_labels: List of predicted class labels for each image
        true_boxes: List of ground truth bounding boxes for each image
        true_labels: List of ground truth class labels for each image
        iou_threshold: IoU threshold for considering a detection as correct
        num_classes: Number of classes
        
    Returns:
        mean Average Precision (mAP)
    """
    aps = []
    
    for class_id in range(num_classes):
        ap = calculate_ap(
            predicted_boxes, predicted_scores, predicted_labels,
            true_boxes, true_labels, class_id, iou_threshold
        )
        aps.append(ap)
    
    return np.mean(aps)


def calculate_metrics(
    predicted_boxes: List[np.ndarray],
    predicted_scores: List[np.ndarray],
    predicted_labels: List[np.ndarray],
    true_boxes: List[np.ndarray],
    true_labels: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive detection metrics.
    
    Args:
        predicted_boxes: List of predicted bounding boxes for each image
        predicted_scores: List of prediction confidence scores for each image  
        predicted_labels: List of predicted class labels for each image
        true_boxes: List of ground truth bounding boxes for each image
        true_labels: List of ground truth class labels for each image
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Dictionary containing various metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_ious = []
    total_hausdorff = []
    
    for i in range(len(predicted_boxes)):
        pred_boxes = predicted_boxes[i] if len(predicted_boxes[i]) > 0 else np.array([]).reshape(0, 4)
        pred_scores = predicted_scores[i] if len(predicted_scores[i]) > 0 else np.array([])
        pred_labels = predicted_labels[i] if len(predicted_labels[i]) > 0 else np.array([])
        gt_boxes = true_boxes[i] if len(true_boxes[i]) > 0 else np.array([]).reshape(0, 4)
        gt_labels = true_labels[i] if len(true_labels[i]) > 0 else np.array([])
        
        # Track which ground truths have been matched
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        # Process each prediction
        for j, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for k, gt_box in enumerate(gt_boxes):
                if not gt_matched[k] and (len(pred_labels) == 0 or len(gt_labels) == 0 or 
                                         pred_labels[j] == gt_labels[k]):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = k
            
            # Determine if prediction is correct
            if best_iou >= iou_threshold:
                total_tp += 1
                gt_matched[best_gt_idx] = True
                total_ious.append(best_iou)
                
                # Calculate Hausdorff distance for correct predictions
                hausdorff_dist = hausdorff_distance(pred_box, gt_boxes[best_gt_idx])
                total_hausdorff.append(hausdorff_dist)
            else:
                total_fp += 1
        
        # Count false negatives (unmatched ground truths)
        total_fn += np.sum(~gt_matched)
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(total_ious) if total_ious else 0.0
    mean_hausdorff = np.mean(total_hausdorff) if total_hausdorff else float('inf')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_iou': mean_iou,
        'mean_hausdorff': mean_hausdorff,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }
