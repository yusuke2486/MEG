import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

def create_mini_batches(features, adj, labels, batch_size):
    data_list = []
    num_nodes = features.size(0)
    
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        batch_indices = torch.arange(start, end)
        # 隣接ノードも含むバッチを作成
        neighbor_indices = torch.cat([batch_indices, adj[batch_indices].nonzero().reshape(-1)])
        neighbor_indices = neighbor_indices.unique()
        
        batch_adj = adj[neighbor_indices][:, neighbor_indices]
        batch_features = features[neighbor_indices]
        batch_labels = labels[neighbor_indices]
        
        edge_index, edge_attr = dense_to_sparse(batch_adj)
        
        data = Data(x=batch_features, edge_index=edge_index, edge_attr=edge_attr, y=batch_labels)
        data_list.append(data)
    
    return data_list

def get_data_loader(features, adj, labels, batch_size):
    data_list = create_mini_batches(features, adj, labels, batch_size)
    return DataLoader(data_list, batch_size=1)  # 各Dataオブジェクトはミニバッチを表します
