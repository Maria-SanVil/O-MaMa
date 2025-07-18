import torch

import torch.nn as nn


class Context_Attn(nn.Module):
    def __init__(self, ch_input, ch_interm, ch_output):
        super(Context_Attn, self).__init__()
        self.n_heads = 4
        self.ch_interm = ch_interm
        self.head_dim = ch_interm // self.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(ch_input, ch_interm)
        self.key = nn.Linear(ch_input, ch_interm)
        self.value = nn.Linear(ch_input, ch_interm)
        self.project = nn.Sequential(nn.Linear(ch_interm, ch_output))
        self.norm_input = nn.LayerNorm(ch_input)
        self.norm_output = nn.LayerNorm(ch_output)
        self.mlp = nn.Sequential(nn.Linear(ch_output, ch_output), nn.GELU(), nn.Linear(ch_output, ch_output))

    def forward(self, desc, img_feat, residual):
        B, N_desc, C = desc.shape  # Batch, Tokens, Channels
        B, N_img, C = img_feat.shape  # Batch, Tokens, Channels
        desc = self.norm_input(desc)
        img_feat = self.norm_input(img_feat)
 
        queries = self.query(desc).view(B, N_desc, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        keys = self.key(img_feat).view(B, N_img, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        values = self.value(img_feat).view(B, N_img, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)

        attn = torch.matmul(queries, keys.transpose(-2, -1))
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        
        y = torch.matmul(attn, values)
        y = y.transpose(1, 2).contiguous().view(B, N_desc, self.ch_interm)
        y = self.project(y)
        if residual:
            y = y + desc
        y = y + self.mlp(self.norm_output(y))
        return y

class MLP(nn.Module):
    def __init__(self, ch_input, ch_intermediate, ch_output):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(ch_input, ch_intermediate), nn.ReLU(), 
                                 nn.Linear(ch_intermediate, ch_intermediate), nn.ReLU(),
                                 nn.Linear(ch_intermediate, ch_output))
    def forward(self, descriptors):
        return self.mlp(descriptors)
    