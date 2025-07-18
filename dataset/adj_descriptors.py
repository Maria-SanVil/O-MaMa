""" Get adjacency matrix for masks using Delaunay triangulation."""

import numpy as np
import torch
from scipy.spatial import Delaunay

def get_adj_matrix(masks, order):
    assert len(masks.shape) == 3, "The mask should have shape (N, H, W)"
    
    N, H, W = masks.shape

    # Get the centroids for each mask
    centroids = []
    for i in range(N):
        coords = (masks[i] > 0).nonzero(as_tuple=False)  # Coordinates of non empty pixels
        if coords.size(0) > 0:  # mask should not be empty
            centroid = coords.float().mean(dim=0)  # Centroid (y, x)
            centroids.append(centroid)

    # Convert to tensor
    if len(centroids) == 0:
        print("Not valid centroids found.")
        centroids = torch.zeros((1), dtype=torch.float32)
    else:
        centroids = torch.stack(centroids)

    # Verify we have enough points for triangulation
    if centroids.size(0) < 3:
        return torch.zeros((N, N), dtype=torch.float32)
    else:    
        centroids_np = centroids.numpy()
        tri = Delaunay(centroids_np)
        
        adj_matrix = np.zeros((N, N), dtype=np.float32)
        indptr, indices = tri.vertex_neighbor_vertices
        for i in range(len(centroids)):
            neighbors = indices[indptr[i]:indptr[i + 1]]
            adj_matrix[i, neighbors] = 1  # Set neighbors to 1
            
        if order == 1:
            adj_matrix = adj_matrix
        elif order == 2:
            adj_matrix_squared = adj_matrix @ adj_matrix
            adj_matrix = adj_matrix + adj_matrix_squared
            adj_matrix = (adj_matrix > 0).astype(int)
        elif order == 3:
            adj_matrix_triple = (adj_matrix @ adj_matrix) @ adj_matrix
            adj_matrix_squared = adj_matrix @ adj_matrix
            adj_matrix = adj_matrix + adj_matrix_squared + adj_matrix_triple
            adj_matrix = (adj_matrix > 0).astype(int)
        adj_matrix = torch.tensor(adj_matrix)
        return adj_matrix   
