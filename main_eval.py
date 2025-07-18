""" Defines the main script for evaluating O-MaMa """

import torch
import argparse

import torch
import numpy as np
import json

from descriptors.get_descriptors import DescriptorExtractor
from dataset.dataset_masks import Masks_Dataset
from model.model import Attention_projector
from evaluation.evaluate import add_to_json, evaluate
from pathlib import Path

import helpers
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def compute_IoU(pred_mask, gt_mask):
    intersection = torch.logical_and(pred_mask, gt_mask).sum()
    union = torch.logical_or(pred_mask, gt_mask).sum()
    IoU = intersection / (union + 1e-6)
    return IoU

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match masks from ego-exo pairs")
    parser.add_argument("--root", type=str, default="/media/maria/Datasets/Ego-Exo4d",help="Path to the dataset")
    parser.add_argument("--reverse", action="store_true", help="Flag to select exo->ego pairs")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size of the dino transformer")
    parser.add_argument("--order", default=2, type=int, help="order of adjacency matrix, 2 for 2nd order")
    parser.add_argument("--context_size", type=int, default=20, help="Size of the context sizo for the object")
    parser.add_argument("--devices", default="0", type=str)
    parser.add_argument("--N_masks_per_batch", default=32, type=int)
    parser.add_argument("--exp_name", type=str, default="Evaluation_OMAMA_Ego->Exo")
    parser.add_argument("--checkpoint_dir", type=str, default="model_weights/best_IoU_Train_OMAMA_ExoEgo.pt")
    args = parser.parse_args()

    helpers.set_all_seeds(42)
    if args.devices != "cpu":
        gpus = [args.devices]  # Specify which GPUs to use
        device_ids = [f'cuda:{gpu}' for gpu in gpus]

        device = torch.device(f'cuda:{device_ids[0].split(":")[1]}') if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    #The test dataset is the full validation dataset, used for the final evaluation
    test_dataset = Masks_Dataset(args.root, args.patch_size, args.reverse, train = False, N_masks_per_batch=args.N_masks_per_batch,order = args.order, test = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 1, pin_memory = True)

    descriptor_extractor = DescriptorExtractor('dinov2_vitb14_reg',  args.patch_size, args.context_size, device)
    model = Attention_projector(args.reverse).to(device)
    checkpoint_weights = torch.load((args.checkpoint_dir))
    model.load_state_dict(checkpoint_weights, strict=False)
    print(model)
    
    processed_test, pred_json_test, gt_json_test = {}, {}, {}
    per_task_processed_test, per_task_pred_json, per_task_gt_json = {}, {}, {}
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        # break
        DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
        SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
        
        is_visible_GT = batch['is_visible']
        POS_mask_position_GT = batch['POS_mask_position']
        similarities, pred_masks_idx, pred_mask, loss, top5_masks = model(SOURCE_descriptors, DEST_descriptors, 
                                                                                                SOURCE_img_feats, DEST_img_feats, 
                                                                                                batch['POS_mask_position'], batch['is_visible'],
                                                                                                batch['DEST_SAM_masks'], test_mode = True)
        
        pred_mask = pred_mask.squeeze().detach().cpu().numpy()
        confidence = similarities.detach().cpu().numpy()  
        
        pred_json_test, gt_json_test = add_to_json(test_dataset, batch['pair_idx'], 
                                                   pred_mask, confidence,
                                                   processed_test, pred_json_test, gt_json_test)

    json_dir = Path('results_json')
    json_dir.mkdir(parents=True, exist_ok=True)
    final_json_gt = {"version": "xx",
                    "challenge": "xx",
                    "annotations": gt_json_test}

    out_dict = evaluate(gt_json_test, pred_json_test, args.reverse)

    #Saving the json with the results
    if args.reverse:
        final_json = {'exo-ego':{'results': pred_json_test}}
        assert "exo-ego" in final_json
        preds = final_json["exo-ego"]

        assert type(preds) == type({})
        for key in ["results"]:
            assert key in preds.keys()

        save_path = os.path.join(json_dir, 'exo2ego_predictions_' + args.exp_name + '.json')
        save_path_gt = os.path.join(json_dir, 'exo2egoGT.json')
    else:
        final_json = {'ego-exo':{'results': pred_json_test}}
        assert "ego-exo" in final_json
        preds = final_json["ego-exo"]

        assert type(preds) == type({})
        for key in ["results"]:
            assert key in preds.keys()
        
        save_path = os.path.join(json_dir, 'ego2exo_predictions_' + args.exp_name + '.json')
        save_path_gt = os.path.join(json_dir, 'ego2exoGT.json')

    with open(save_path, 'w') as f:
        json.dump(convert_ndarray(final_json), f)

    with open(save_path_gt, 'w') as f:
        json.dump(convert_ndarray(final_json_gt), f)
    
    