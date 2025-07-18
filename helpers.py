import os 
import numpy as np 
import torch
import random


def set_all_seeds(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    random.seed(seed)    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def our_collate_fn(batch):
    return {key: torch.stack([sample[key] for sample in batch]) for key in batch[0]}

