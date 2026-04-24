import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx


class ThesisMacroMetrics(nn.Module):
    """
    Computes macro graph topology metrics used in the previous thesis baseline:
    1. Average Subgraph Size (Connected Components)
    2. Average Intra-cluster Distance (Average shortest path within components)
    3. Modularity (using Louvain)
    4. Average Clustering Coefficient
    5. Num nodes
    6. Num edges 
    7. Density (for undirected graphs: edges / (nodes * (nodes - 1) / 2))
    Stores the result in data.macro_metrics (Tensor of shape [1, 7])
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, data):
        device = data.x.device if data.x is not None else 'cpu'
        G = to_networkx(data, to_undirected=True)
        
        components = list(nx.connected_components(G))
        if len(components) > 0:
            avg_subgraph_size = np.mean([len(c) for c in components])
        else:
            avg_subgraph_size = 0.0
            
        intra_dists = []
        for c in components:
            if len(c) > 1:
                subg = G.subgraph(c)
                try:
                    intra_dists.append(nx.average_shortest_path_length(subg))
                except:
                    pass
        avg_intra_dist = np.mean(intra_dists) if len(intra_dists) > 0 else 0.0
        
        try:
            communities = nx.community.louvain_communities(G)
            modularity = nx.community.modularity(G, communities)
        except:
            modularity = 0.0
            
        try:
            clustering_coeff = nx.average_clustering(G)
        except:
            clustering_coeff = 0.0

        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1] / 2
        
        if num_nodes > 1:
            density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
        else:
            density = 0.0
            
        metrics = [
            avg_subgraph_size,
            avg_intra_dist,
            modularity,
            clustering_coeff,
            num_nodes,
            num_edges,
            density,
        ]
        data.macro_metrics = torch.tensor(metrics, dtype=torch.float32, device=device).unsqueeze(0)
        return data
