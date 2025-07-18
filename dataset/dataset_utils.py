""" Utility functions for computing IoU and bounding boxes from masks."""

import torch
import numpy as np

# This function computes the Intersection over Union (IoU) between a set of SAM masks and a ground truth mask.
def compute_IoU(SAM_masks, mask_GT):
    SAM_masks = SAM_masks.bool()
    mask_GT = mask_GT.bool()

    intersection = SAM_masks & mask_GT
    union = SAM_masks | mask_GT
    iou = intersection.sum(dim=(1,2)).float() / (union.sum(dim=(1,2)).float() + 1e-6)
    return iou    
    
# This function computes the Intersection over Union (IoU) between a set of bounding boxes (SAM_bboxes) and a ground truth bounding box (bbox_GT).
def compute_IoU_bbox(SAM_bboxes, bbox_GT):
    # BBox format is (x1, y1, w, h)
    # Ensure proper format for SAM_bboxes and bbox_GT
    sam_xmin = SAM_bboxes[:, 0]
    sam_ymin = SAM_bboxes[:, 1]
    sam_xmax = SAM_bboxes[:, 0] + SAM_bboxes[:, 2]  # x1 + w
    sam_ymax = SAM_bboxes[:, 1] + SAM_bboxes[:, 3]  # y1 + h
        
    gt_xmin = bbox_GT[0]
    gt_ymin = bbox_GT[1]
    gt_xmax = bbox_GT[0] + bbox_GT[2]  # x1 + w
    gt_ymax = bbox_GT[1] + bbox_GT[3]  # y1 + h

    # Calculate intersection bounds
    inter_xmin = torch.max(sam_xmin, gt_xmin)
    inter_ymin = torch.max(sam_ymin, gt_ymin)
    inter_xmax = torch.min(sam_xmax, gt_xmax)
    inter_ymax = torch.min(sam_ymax, gt_ymax)

    # Calculate intersection area
    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_width * inter_height

    # Calculate areas of SAM_bboxes and bbox_GT
    sam_area = SAM_bboxes[:, 2] * SAM_bboxes[:, 3]  # w * h
    gt_area = bbox_GT[2] * bbox_GT[3]              # w * h

    # Calculate union area
    union_area = sam_area + gt_area - inter_area

    # Compute IoU
    iou = inter_area / union_area
    iou = torch.clamp(iou, min=0, max=1)  # Clamp IoU to valid range
    return iou

# This function computes the bounding box from a binary mask.
def bbox_from_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.to(dtype=torch.bool)  # Ensure boolean dtype
        if not mask.any():
            return torch.zeros(4, dtype=torch.float32, device=mask.device), False
        
        y_min, y_max = mask.nonzero(as_tuple=True)[0].min(), mask.nonzero(as_tuple=True)[0].max()
        x_min, x_max = mask.nonzero(as_tuple=True)[1].min(), mask.nonzero(as_tuple=True)[1].max()
        
        return torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min], dtype=torch.float32, device=mask.device), True

    elif isinstance(mask, np.ndarray):
        if not mask.any():
            return np.zeros(4, dtype=np.float32), False
        
        y, x = np.where(mask)
        return np.array([x.min(), y.min(), x.max() - x.min(), y.max() - y.min()], dtype=np.float32), True

    else:
        raise ValueError("Unsupported mask type. Expected torch.Tensor or np.ndarray.")
