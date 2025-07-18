""" Dataloader for the Ego-Exo4D correspondences dataset """

import os 
import torch
import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch
from torch.utils.data import Dataset
import random

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset.adj_descriptors import get_adj_matrix
from dataset.dataset_utils import compute_IoU, compute_IoU_bbox, bbox_from_mask


class Masks_Dataset(Dataset):
    def __init__(self, root, patch_size, reverse, N_masks_per_batch, order, train, test):
        self.root = root
        self.train_mode = train
        self.test_mode = test
        self.reverse = reverse

        # Select the pre-extracted masks directory based on the train/test mode and reverse flag
        if train:
            if reverse:
                self.masks_dir = os.path.join(root, 'Masks_TRAIN_EXO2EGO')
            else:
                self.masks_dir = os.path.join(root, 'Masks_TRAIN_EGO2EXO')
        else:
            if test:
                if reverse:
                    self.masks_dir = os.path.join(root, 'Masks_TEST_EXO2EGO')
                else:
                    self.masks_dir = os.path.join(root, 'Masks_TEST_EGO2EXO')
                    
            else:
                if reverse:
                    self.masks_dir = os.path.join(root, 'Masks_VAL_EXO2EGO')
                else:   
                    self.masks_dir = os.path.join(root, 'Masks_VAL_EGO2EXO')

        # Preprocessed dataset directory
        self.dataset_dir = os.path.join(root, 'processed')

        # Configs for loading the features
        self.N_masks_per_batch = N_masks_per_batch
        self.patch_size = patch_size

        self.order = order

        # Load the mask annotations and pairs
        self.mask_annotations = self.load_mask_annotations()
        self.pairs = self.load_all_pairs()
        self.takes_json = json.load(open(os.path.join(root, 'takes.json'), 'r'))

        # Transformations for the images
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

        print(len(self.takes_json), 'TAKES')

    # Load the json with the pairs
    def load_all_pairs(self):
        # Select the json file based on the train/test mode and reverse flag
        if self.train_mode:
            if self.reverse:
                pairs_json = 'train_exoego_20percent_pairs.json' # We train with 20% of the pairs of the full trainig set
            else:
                pairs_json = 'train_egoexo_20percent_pairs.json' # We train with 20% of the pairs of the full trainig set
        else:
            if self.test_mode:
                if self.reverse:
                    pairs_json = 'test_exoego_pairs.json'
                else:
                    pairs_json = 'test_egoexo_pairs.json'
            
            #Validation is just a subset of the test
            else:    
                if self.reverse:
                    pairs_json = 'custom_val_exoego_pairs_10.json' # We validate with 10% of the pairs of the full validation set
                else:
                    pairs_json = 'custom_val_egoexo_pairs_10.json' # We validate with 10% of the pairs of the full validation set

        print('----------------------------We are loading: ', pairs_json, 'with the pair of images')
        pairs = []
        jsons_dir = os.path.join(self.root,'dataset_jsons') # Put the jsons generated from the data preparation here
        with open(os.path.join(jsons_dir, pairs_json), 'r') as fp:
            pairs.extend(json.load(fp))
        print('LEN OF THE DATASET:', len(pairs))
        return pairs
    
    # Load the GT mask annotations
    def load_mask_annotations(self):
        d = self.dataset_dir
        with open(f'{d}/split.json', 'r') as fp:
            splits = json.load(fp)
        valid_takes = splits['train'] + splits['val'] + splits['test']

        annotations = {}
        for take in valid_takes:
            try:
                with open(f'{d}/{take}/annotation.json', 'r') as fp:
                    annotations[take] = json.load(fp)
            except:
                continue
        return annotations

    # Returns the img reshaped slightly to be divisible by 14 (dinov2 patch size)
    def reshape_img(self, img):
        h, w = img.shape[:2]
        new_h = self.patch_size * (h // self.patch_size)
        new_w = self.patch_size * (w // self.patch_size)
        size = (new_w, new_h)
        if h % self.patch_size != 0:
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        return img

    def check_horizontal(self, img):
        h, w = img.shape[:2]
        if h > w:
            horiz_size = (h, w)
            img = cv2.resize(img, horiz_size, interpolation=cv2.INTER_NEAREST)
        return img

    # Select the adjacent negatives based on the adjacency matrix   
    def select_adjacent_negatives(self, adj_matrix, SAM_bboxes, SAM_masks, mask_GT):
        # Select adjacent negatives based on the adjacency matrix
        
        bbox_GT, _ = bbox_from_mask(mask_GT)
        bbox_iou = compute_IoU_bbox(SAM_bboxes, bbox_GT)
        max_index = torch.argmax(bbox_iou)
        
        # Get the neighbors of the best mask
        adj_matrix[max_index, max_index] = 0
        neighbors = torch.where(adj_matrix[max_index] == 1)[0]
        N_adjacent_indices = self.N_masks_per_batch - 1
        if len(neighbors) > N_adjacent_indices:
            random_indices = np.random.choice(neighbors, N_adjacent_indices, replace=False)
            adjacent_SAM_masks = SAM_masks[random_indices]
            adjacent_SAM_bboxes = SAM_bboxes[random_indices]
        else:
            adjacent_SAM_masks = SAM_masks[neighbors]
            adjacent_SAM_bboxes = SAM_bboxes[neighbors]
            
            # Get remaining negatives
            N_remaining_indices = N_adjacent_indices - len(neighbors)
            if SAM_masks.shape[0] < N_remaining_indices:
                remaining_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=True)
            else:
                remaining_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=False)
                
            
            adjacent_SAM_masks = torch.cat((adjacent_SAM_masks, SAM_masks[remaining_indices]), dim=0)
            adjacent_SAM_bboxes = torch.cat((adjacent_SAM_bboxes, SAM_bboxes[remaining_indices]), dim=0)
            
        
        return adjacent_SAM_masks, adjacent_SAM_bboxes

    # Select the best SAM mask
    def get_best_mask(self, SAM_masks, mask_GT):
        iou = compute_IoU(SAM_masks, mask_GT)
        return torch.argmax(iou)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx_sample):
        # Get the pair of images, 1 refers to the source image and 2 to the destination image
        if self.reverse:
            # img_pth2 ego, img_pth1 exo
            img_pth2, _, img_pth1, _ = self.pairs[idx_sample]
        else:
            # img_pth2 exo, img_pth1 ego
            img_pth1, _, img_pth2, _ = self.pairs[idx_sample]

        root, take_id, cam, obj, _, idx = img_pth1.split('//')
        root = self.dataset_dir
        root2, take_id2, cam2, obj2, _, idx2 = img_pth2.split('//')
        root2 = self.dataset_dir

        # Both viewpoints should have the same take_id, object and index   
        assert obj == obj2
        assert idx == idx2
        assert take_id == take_id2

        vid_idx = int(idx)
        vid_idx2 = int(idx2)

        # Image 1: SOURCE
        img1 = cv2.imread(f"{root}/{take_id}/{cam}/{vid_idx}.jpg")[..., ::-1]
        if self.train_mode:
            img1 = self.check_horizontal(img1)
        img1 = self.reshape_img(img1)
        self.h1, self.w1 = img1.shape[:2]
        img1_torch = self.transform_img(img1)

        # Load the source mask
        mask_annotation_SOURCE = self.mask_annotations[take_id]
        mask_SOURCE = mask_utils.decode(mask_annotation_SOURCE['masks'][obj][cam][idx])
        mask_SOURCE = cv2.resize(mask_SOURCE, (img1.shape[1],img1.shape[0]), interpolation=cv2.INTER_NEAREST)
        assert mask_SOURCE.shape == img1.shape[:2]
        mask_SOURCE = torch.from_numpy(mask_SOURCE.astype(np.uint8))
        
        
        # Image 2: DESTINATION
        img2 = cv2.imread(f"{root2}/{take_id2}/{cam2}/{vid_idx2}.jpg")[..., ::-1]
        if self.train_mode:
            img2 = self.check_horizontal(img2)
        img2 = self.reshape_img(img2)
        self.h2, self.w2 = img2.shape[:2]
        img2_torch = self.transform_img(img2)

                
        # Load the destination GT mask
        mask_annotation_DEST = self.mask_annotations[take_id2]
        if idx in mask_annotation_DEST['masks'][obj2][cam2]:  # If the object is visible in the destionation image, load it
            mask2_GT = mask_utils.decode(mask_annotation_DEST['masks'][obj2][cam2][idx])
            mask2_GT = cv2.resize(mask2_GT, (img2.shape[1],img2.shape[0]), interpolation=cv2.INTER_NEAREST)
        else: # If the object is not visible in the destination image, create an empty mask
            mask2_GT = np.zeros(img2.shape[:2])
        assert mask2_GT.shape == img2.shape[:2]
        mask2_GT = torch.from_numpy(mask2_GT.astype(np.uint8))

        # Load the proposed pre-extracted SAM masks for this pair
        SAM_masks = np.load(f"{self.masks_dir}/{take_id2}/{cam2}/{vid_idx2}_masks.npz")
        SAM_masks = torch.from_numpy(SAM_masks['arr_0'].astype(np.uint8)) # N, H, W. H = 532, W = 952
        if len(SAM_masks.shape) < 3:
            SAM_masks = torch.zeros((1, self.h2, self.w2))
        N_masks, H_masks, W_masks = SAM_masks.shape
        if H_masks != self.h2 or W_masks != self.w2: # Only in inference for FastSAM masks
            SAM_masks = F.interpolate(SAM_masks.unsqueeze(0).float(), size=(self.h2, self.w2), mode='nearest').squeeze(0).long()
        
        # Get the adjacent matrix
        adj_matrix = get_adj_matrix(SAM_masks, order=self.order)
        
        SAM_bboxes_dest = np.load(f"{self.masks_dir}/{take_id2}/{cam2}/{vid_idx2}_boxes.npy")
        SAM_bboxes_dest = torch.from_numpy(SAM_bboxes_dest.astype(np.float32)) # x1, y1, w, h
        h_factor = self.h2 / H_masks
        w_factor = self.w2 / W_masks
        if h_factor != 1 or w_factor != 1:
            SAM_bboxes_dest[:, 0] = SAM_bboxes_dest[:, 0] * w_factor
            SAM_bboxes_dest[:, 1] = SAM_bboxes_dest[:, 1] * h_factor
            SAM_bboxes_dest[:, 2] = SAM_bboxes_dest[:, 2] * w_factor
            SAM_bboxes_dest[:, 3] = SAM_bboxes_dest[:, 3] * h_factor       

        if self.train_mode:
            visible_pixels = mask2_GT.sum()
            # If the object is visible in the destination image, we select the best SAM mask as the positive mask
            if visible_pixels > 0:
                NEG_SAM_masks, NEG_SAM_bboxes = self.select_adjacent_negatives(adj_matrix, SAM_bboxes_dest, SAM_masks, mask2_GT)
                is_visible = torch.tensor(1.) # True
                POS_SAM_masks = mask2_GT # Choose the GT (Strong Positive) or the best SAM mask (Weak Positive) TODO
                POS_SAM_bboxes, _ = bbox_from_mask(mask2_GT) # x1, y1, w, h
            else:
                N_remaining_indices = self.N_masks_per_batch - 1
                if SAM_masks.shape[0] < N_remaining_indices:
                    random_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=True)
                else:
                    random_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=False)
                
                NEG_SAM_masks = SAM_masks[random_indices]
                NEG_SAM_bboxes = SAM_bboxes_dest[random_indices]
                is_visible = torch.tensor(0.) #False
                random_idx = np.random.randint(SAM_masks.shape[0])
                POS_SAM_masks = SAM_masks[random_idx]
                POS_SAM_bboxes = SAM_bboxes_dest[random_idx]

            POS_mask_position = random.randint(0, self.N_masks_per_batch - 1) # Random position of the positive mask in the batch
            NEG_part1 = NEG_SAM_masks[:POS_mask_position]
            NEG_part2 = NEG_SAM_masks[POS_mask_position:]
            DEST_SAM_masks = torch.cat((NEG_part1, POS_SAM_masks.unsqueeze(0), NEG_part2), dim=0)

            NEG_part1_bboxes = NEG_SAM_bboxes[:POS_mask_position]
            NEG_part2_bboxes = NEG_SAM_bboxes[POS_mask_position:]
            DEST_SAM_bboxes = torch.cat((NEG_part1_bboxes, POS_SAM_bboxes.unsqueeze(0), NEG_part2_bboxes), dim=0)

        # In validation or test modes, we just return the SAM masks, and precompute which is the best SAM mask
        else:
            DEST_SAM_masks = SAM_masks
            visible_pixels = mask2_GT.sum()
            if visible_pixels > 0:
                is_visible = torch.tensor(1.) # True
            else:
                is_visible = torch.tensor(0.) # False
            POS_mask_position = self.get_best_mask(SAM_masks, mask2_GT)
            DEST_SAM_bboxes = SAM_bboxes_dest
            if len(DEST_SAM_bboxes.shape) == 1:
                DEST_SAM_bboxes = torch.zeros((1, 4))
        
        return {'SOURCE_img': img1_torch, 'SOURCE_mask': mask_SOURCE, 'SOURCE_bbox': bbox_from_mask(mask_SOURCE)[0], 'SOURCE_img_size': torch.tensor([self.h1, self.w1]),
                'GT_img': img2_torch, 'GT_mask': mask2_GT, 
                'DEST_SAM_masks': DEST_SAM_masks, 'DEST_SAM_bbox': DEST_SAM_bboxes, 'DEST_img_size': torch.tensor([self.h2, self.w2]),
                'is_visible': is_visible, 'POS_mask_position': torch.tensor(POS_mask_position),
                'pair_idx': torch.tensor(idx_sample)}






