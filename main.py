""" Defines the main script for training O-MaMa """

import torch
import argparse
from descriptors.get_descriptors import DescriptorExtractor
from dataset.dataset_masks import Masks_Dataset
from model.model import Attention_projector
from evaluation.evaluate import add_to_json, evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

import helpers
from tqdm import tqdm
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match masks from ego-exo pairs")
    parser.add_argument("--root", type=str, default="/media/maria/Datasets/Ego-Exo4d",help="Path to the dataset")
    parser.add_argument("--reverse", action="store_true", help="Flag to select exo->ego pairs")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size of the dino transformer")
    parser.add_argument("--context_size", type=int, default=20, help="Size of the context sizo for the object")
    parser.add_argument("--devices", default="0", type=str)
    parser.add_argument("--N_masks_per_batch", default=32, type=int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--N_epochs", default=3, type=int)
    parser.add_argument("--order", default=2, type=int, help="order of adjacency matrix, 2 for 2nd order")
    parser.add_argument("--exp_name", type=str, default="Train_OMAMA_EgoExo")
    args = parser.parse_args()

    helpers.set_all_seeds(42)
    if args.devices != "cpu":
        gpus = [args.devices]  # Specify which GPUs to use
        device_ids = [f'cuda:{gpu}' for gpu in gpus]

        device = torch.device(f'cuda:{device_ids[0].split(":")[1]}') if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    # Training dataset only contains horizontal images, in order to batchify the masks
    train_dataset = Masks_Dataset(args.root, args.patch_size, args.reverse, N_masks_per_batch=args.N_masks_per_batch, order = args.order, train = True, test = False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=helpers.our_collate_fn, num_workers = 8, pin_memory = True) #16 in both
    
    # Validation dataset contains both horizontal and vertical images. Since the SAM masks are different, we use batch size 1
    # Note: the val annotations are a small subset of the full validation dataset, used for eval the training per epoch
    val_dataset = Masks_Dataset(args.root, args.patch_size, args.reverse, args.N_masks_per_batch,  order = args.order, train = False, test = False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    best_IoU = 0

    descriptor_extractor = DescriptorExtractor('dinov2_vitb14_reg', args.patch_size, args.context_size, device)
    model = Attention_projector(reverse = args.reverse).to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    T_max = args.N_epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min=1e-6)

    folder_weights = Path("model_weights")
    folder_weights.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.N_epochs):
        print('-----------------------Starting epoch training-----------------------')
        epoch_loss = 0
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
            SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
            best_similarities, best_masks, refined_mask, loss, top5_masks = model(SOURCE_descriptors, DEST_descriptors, 
                                                                                  SOURCE_img_feats, DEST_img_feats, 
                                                                                  batch['POS_mask_position'], batch['is_visible'],
                                                                                  batch['DEST_SAM_masks'], test_mode = False)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        
        epoch_loss /= len(train_dataloader)
        print('--------------------------Epoch ', epoch, ' loss: ', epoch_loss, '--------------------------')
        torch.save(model.state_dict(), os.path.join(folder_weights, 'last_epoch' + '_' + args.exp_name + '.pt'))
        
        print('-----------------------Starting epoch validation-----------------------')
        # Validation loop
        processed_epoch, pred_json_epoch, gt_json_epoch = {}, {}, {}
        epoch_loss = 0
        model.eval()
        for idx, batch in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
                SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
                similarities, pred_masks_idx, refined_mask, loss, top5_masks = model(SOURCE_descriptors, DEST_descriptors, 
                                                                                     SOURCE_img_feats, DEST_img_feats, 
                                                                                     batch['POS_mask_position'], batch['is_visible'],
                                                                                     batch['DEST_SAM_masks'], test_mode = False)
                pred_mask = refined_mask.squeeze().detach().cpu().numpy()
                confidence = similarities.detach().cpu().numpy()
                
                epoch_loss += loss.item()
                pred_json_epoch, gt_json_epoch = add_to_json(val_dataset, batch['pair_idx'], 
                                                            pred_mask, confidence,
                                                            processed_epoch, pred_json_epoch, gt_json_epoch)

        epoch_loss /= len(val_dataloader)
        print('--------------------------Epoch ', epoch, ' metrics--------------------------')
        out_dict = evaluate(gt_json_epoch, pred_json_epoch, args.reverse)  
        if out_dict['iou'] > best_IoU:
            best_IoU = out_dict['iou']
            torch.save(model.state_dict(), os.path.join('model_weights', 'best_IoU' + '_' + args.exp_name + '.pt'))
