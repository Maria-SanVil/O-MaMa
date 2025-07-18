""" This extracts DINO descriptors of each mask, and outputs the batch with positive and negative pairs """

import torch

class DescriptorExtractor:
    def __init__(self, dino_model, patch_size, context_size, device):
        # Load DINOv2 pre-trained model 
        self.model = torch.hub.load('facebookresearch/dinov2', dino_model).to(device)
        self.model.eval()
        self.patch_size = patch_size
        self.device = device
        self.context_size = context_size

    def extract_dense_features(self, image_tensor):
        with torch.no_grad():  # Disable gradient computation
            B, _, h, w = image_tensor.shape
            features = self.model.forward_features(image_tensor)
            patch_tokens = features['x_norm_patchtokens'].detach()
            patch_tokens = patch_tokens.reshape((B, h // self.patch_size, w // self.patch_size, patch_tokens.shape[-1])).permute(0, 3, 1, 2)
        return patch_tokens.to(self.device)
       
    def dense_mask(self, masks, features):
        B, Nmasks, _, _ = masks.shape
        descriptor_list = torch.zeros((B, Nmasks, features.shape[1]), device=features.device)
        for b in range(masks.shape[0]):
            masks_expanded_batch = torch.nn.functional.interpolate(masks[b].unsqueeze(0).float(),
                                                                size = (features.shape[2], features.shape[3]),
                                                                mode='nearest').squeeze(0).to(self.device)
            for m in range(masks_expanded_batch.shape[0]):
                mask_expanded = masks_expanded_batch[m].unsqueeze(0)
                mask_descriptor = features[b] * mask_expanded
                mask_sum = mask_expanded.sum(dim=(1, 2), keepdim=True)  # Nº of píxeles with active mask per channel
                if mask_sum != 0:
                    feature_mean = (mask_descriptor.sum(dim=(1, 2)) / mask_sum.squeeze(0)).nan_to_num(0)
                    descriptor_list[b, m] = feature_mean
        return descriptor_list
     
    def add_context_to_bbox(self, bboxes_masks, context_size, H_max, W_max, reduction_factor):
        bboxes_context = bboxes_masks.clone().to(torch.int32) # Format x1, y1, w, h
        
        #Convert to x1, y1, x2, y2
        bboxes_context[:, :, 2] = bboxes_context[:, :, 0] + bboxes_context[:, :, 2]
        bboxes_context[:, :, 3] = bboxes_context[:, :, 1] + bboxes_context[:, :, 3]

        # Add context and validate limits
        bboxes_context[:, :, 0] = torch.clamp(bboxes_context[:, :, 0] - context_size, 0, W_max)  # x1 - context
        bboxes_context[:, :, 1] = torch.clamp(bboxes_context[:, :, 1] - context_size, 0, H_max)  # y1 - context
        bboxes_context[:, :, 2] = torch.clamp(bboxes_context[:, :, 2] + context_size, 0, W_max)  # x2 + context
        bboxes_context[:, :, 3] = torch.clamp(bboxes_context[:, :, 3] + context_size, 0, H_max)  # y2 + context

        bboxes_context = torch.floor(bboxes_context / reduction_factor).int()
        bboxes_context[:, :, 2] = torch.max(bboxes_context[:, :, 0] + 1, bboxes_context[:, :, 2])
        bboxes_context[:, :, 3] = torch.max(bboxes_context[:, :, 1] + 1, bboxes_context[:, :, 3])

        # Limit coordinates to the grid
        max_x = W_max // reduction_factor - 1
        max_y = H_max // reduction_factor - 1
        bboxes_context[:, :, 0] = torch.clamp(bboxes_context[:, :, 0], 0, max_x)  # x1
        bboxes_context[:, :, 1] = torch.clamp(bboxes_context[:, :, 1], 0, max_y)  # y1
        bboxes_context[:, :, 2] = torch.clamp(bboxes_context[:, :, 2], 0, max_x + 1)  # x2
        bboxes_context[:, :, 3] = torch.clamp(bboxes_context[:, :, 3], 0, max_y + 1)  # y2
        return bboxes_context

    def dense_bbox(self, bboxes_masks, img_sizes, features, context_sizes, reduction_factor):
        _, C, _, _ = features.shape
        H_max = img_sizes[:, 0].max().item()
        W_max = img_sizes[:, 1].max().item()
        descriptors = []

        # Iterate through batches
        for b in range(features.shape[0]):
            batch_descriptors = []

            # Process each context size
            for context_size in context_sizes:
                # Add context to bounding boxes for the current size
                bboxes_context = self.add_context_to_bbox(bboxes_masks[b].unsqueeze(0), context_size, H_max, W_max, reduction_factor)

                # Initialize the descriptor list for the current context size
                descriptor_list = torch.zeros((bboxes_context.shape[1], C), device=features.device)
                # Iterate through bounding boxes
                for i in range(bboxes_context.shape[1]):
                    x1, y1, x2, y2 = bboxes_context[0, i, :]
                    mask_descriptors = features[b, :, int(y1.item()):int(y2.item()), int(x1.item()):int(x2.item())]
                    mean_descriptor = mask_descriptors.mean(dim=(1, 2))
                    
                    descriptor_list[i] = mean_descriptor
                    
                batch_descriptors.append(descriptor_list)

            # Concatenate descriptors for all context sizes for this batch
            descriptors.append(torch.cat(batch_descriptors, dim=1))

        # Stack all batch descriptors
        descriptor_list = torch.stack(descriptors)
        return descriptor_list
    
    # The DEST descriptors are N masks, with negatives and a positive pair
    def get_DEST_descriptors(self, batch):
        DEST_img = batch['GT_img'].to(self.device)

        feats_DEST_img = self.extract_dense_features(DEST_img)
        
        _, _, h, w = feats_DEST_img.shape
        reduction_factor = 4
        dense_features = torch.nn.functional.interpolate(feats_DEST_img, 
                                                            size = (int(h * self.patch_size / reduction_factor), int(w * self.patch_size / reduction_factor)), 
                                                            mode='bilinear', align_corners=False)
        context_DEST_descriptors = self.dense_bbox(batch['DEST_SAM_bbox'], batch['DEST_img_size'], dense_features, context_sizes=[100], reduction_factor=reduction_factor)
        mask_DEST_descriptors = self.dense_mask(batch['DEST_SAM_masks'], dense_features)
        DEST_descriptors = torch.cat((mask_DEST_descriptors, context_DEST_descriptors), dim=2).to(self.device)

        return DEST_descriptors.to(self.device), feats_DEST_img

    # The SOURCE descriptors are just one mask 
    def get_SOURCE_descriptors(self, batch):
        SOURCE_img = batch['SOURCE_img'].to(self.device)

        feats_SOURCE_img = self.extract_dense_features(SOURCE_img)

        _, _, h, w = feats_SOURCE_img.shape
        reduction_factor = 4
        dense_features = torch.nn.functional.interpolate(feats_SOURCE_img, 
                                                            size = (int(h * self.patch_size / reduction_factor), int(w * self.patch_size / reduction_factor)), 
                                                            mode='bilinear', align_corners=False)
        context_SOURCE_descriptors = self.dense_bbox(batch['SOURCE_bbox'].unsqueeze(1), batch['SOURCE_img_size'], dense_features, context_sizes=[100], reduction_factor=reduction_factor)
        mask_SOURCE_descriptors = self.dense_mask(batch['SOURCE_mask'].unsqueeze(1), dense_features)
        SOURCE_descriptors = torch.cat((mask_SOURCE_descriptors, context_SOURCE_descriptors), dim=2).to(self.device)
        
        return SOURCE_descriptors.to(self.device), feats_SOURCE_img
    