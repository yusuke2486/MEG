import torch
import torch.nn as nn

class AdjacencyGenerator(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, hidden_size, device, dropout=0.1):
        super(AdjacencyGenerator, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 2).to(device)
        self.linear2 = nn.Linear(d_model * 2, 1).to(device)
        self.device = device

    def forward(self, node_features, neighbor_features):
        # node_features shape: (1, d_model)
        # neighbor_features shape: (num_neighbors, d_model)
        
        # Concatenate node_features and neighbor_features
        input_features = torch.cat([node_features, neighbor_features], dim=0).to(self.device)  # (num_neighbors + 1, d_model)
        # print(f"input_features.shape: {input_features.shape}")
        
        # Pass through linear layers
        hidden = torch.relu(self.linear1(input_features))  # (num_neighbors + 1, d_model * 2)
        # print(f"hidden.shape: {hidden.shape}")
        adj_logits = self.linear2(hidden).squeeze(-1)  # (num_neighbors + 1)
        # print(f"adj_logits.shape: {adj_logits.shape}")

        # Apply logistic sigmoid to get probabilities
        adj_probs = torch.sigmoid(adj_logits).to(self.device)  # (num_neighbors + 1)
        # print(f"adj_probs.shape: {adj_probs.shape}")

        return adj_probs

    def generate_new_neighbors(self, node_features, neighbor_features):
        adj_probs = self.forward(node_features, neighbor_features)
        new_neighbors = torch.bernoulli(adj_probs).to(self.device)  # Sample new neighbors
        # print(f"adj_probs: {adj_probs}")
        # print(f"adj_probs.shape: {adj_probs.shape}")
        # print(f"new_neighbors: {new_neighbors}")
        # print(f"new_neighbors.shape: {new_neighbors.shape}")

        return adj_probs, new_neighbors
