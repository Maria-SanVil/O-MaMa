import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from model.model_layers import Context_Attn, MLP
    
class Attention_projector(nn.Module):
    def __init__(self, reverse, ch_input = 768, ch_intermediate = 768, ch_output = 768):
        super(Attention_projector, self).__init__()
        self.mlp = MLP(3 * ch_input, ch_intermediate, ch_output)
        self.mlp_visible = nn.Sequential(nn.Linear(ch_output * 2, ch_output), nn.ReLU(),
                                         nn.Linear(ch_output, 256), nn.ReLU(),
                                         nn.Linear(256, 128), nn.ReLU(),
                                         nn.Linear(128, 2))
        
        #------------------------Cross Attention module
        self.CROSS_context_attn = Context_Attn(ch_input = 768, ch_interm = 768, ch_output = 768)
        if reverse:
            self.pos_embed_T = nn.Parameter(torch.zeros(1, 50*50, ch_input))
            self.pos_embed_Q = nn.Parameter(torch.zeros(1, 38*68, ch_input))
        else:
            self.pos_embed_T = nn.Parameter(torch.zeros(1, 38*68, ch_input))
            self.pos_embed_Q = nn.Parameter(torch.zeros(1, 50*50, ch_input))
        self.init_weights()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def init_weights(self):
        trunc_normal_(self.pos_embed_T, std=0.02)
        trunc_normal_(self.pos_embed_Q, std=0.02)
    
    def forward(self, source_descriptors, dest_descriptors, source_dense_feats, dest_dense_feats, GT_position, GT_visible, source_descriptors_SAM_masks, test_mode):
        source_obj = source_descriptors[:, :, :768]
        dest_obj = dest_descriptors[:, :, :768]

        pos_embed_T = self.pos_embed_T.to(dest_dense_feats.device)
        dest_dense_feats = dest_dense_feats.flatten(2).transpose(1, 2)
        dest_dense_feats = dest_dense_feats + pos_embed_T
        Q_context_cross = self.CROSS_context_attn(source_obj, dest_dense_feats, residual = False)

        source_dense_feats = source_dense_feats.flatten(2).transpose(1, 2)
        source_dense_feats = source_dense_feats + self.pos_embed_Q.to(source_dense_feats.device)
        T_context_cross = self.CROSS_context_attn(dest_obj, source_dense_feats, residual = False)
        
        Q_desc = self.mlp(torch.cat([Q_context_cross, source_descriptors], dim=2))
        T_desc = self.mlp(torch.cat([T_context_cross, dest_descriptors], dim=2))

        Q_desc_norm = F.normalize(Q_desc, p=2, dim=2)
        T_desc_norm = F.normalize(T_desc, p=2, dim=2)
    
        similarity = F.cosine_similarity(Q_desc_norm, T_desc_norm, dim=2).unsqueeze(1)
        B, Nq, Nt = similarity.shape
        similarity = similarity.reshape(B*Nq, Nt)
        best_similarities, best_masks = torch.max(similarity, dim=1)
        if similarity.shape[1] < 5:
            topk_indices = best_masks.unsqueeze(1).expand(-1, 5)
        else:
            _, topk_indices = torch.topk(similarity, k=5, dim=1)
                
        full_pred_mask = source_descriptors_SAM_masks[0, best_masks[0].item(), :, :]
        
        if test_mode:
            total_loss = None
        else:
            gt_idx_flat = GT_position.reshape(B*Nq).to(Q_desc.device)
            loss = self.cross_entropy(similarity / 0.07, gt_idx_flat)
            loss_mask = (loss * GT_visible.to(Q_desc.device)).mean()

            total_loss = loss_mask
        best_similarities = torch.sigmoid(best_similarities)
        return best_similarities, best_masks, full_pred_mask, total_loss, topk_indices
