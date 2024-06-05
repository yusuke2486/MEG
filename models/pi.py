import torch
import torch.nn as nn
from models.transformer import TransformerEncoder

class AdjacencyGenerator(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, hidden_size, device, dropout=0.1):
        super(AdjacencyGenerator, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout).to(device)
        self.device = device

        # Linear layer to reduce adj_logits to a single value for each node
        self.weight_vector = nn.Linear(d_model, 1).to(device)

        # # Initialize weights to 0
        # self._initialize_weights()

    def _initialize_weights(self):
        for p in self.encoder.parameters():
            nn.init.constant_(p, 0)
        nn.init.constant_(self.weight_vector.weight, 0)
        nn.init.constant_(self.weight_vector.bias, 0)

    def forward(self, node_features, neighbor_features):
        input_features = torch.cat([node_features, neighbor_features], dim=0)  # (num_neighbors + 1, 1, d_model)        
        adj_logits = self.encoder(input_features)  # (num_neighbors + 1, 1, d_model)        
        adj_logits = adj_logits  # (num_neighbors + 1, d_model)        
        adj_logits = self.weight_vector(adj_logits).squeeze(1)  # (num_neighbors + 1)
        adj_probs = torch.sigmoid(adj_logits/3).to(self.device)  # Reduce to (num_neighbors + 1)

        return adj_probs, adj_logits

    def generate_new_neighbors(self, node_features, neighbor_features):
        adj_probs, adj_logits = self.forward(node_features, neighbor_features)
        new_neighbors = torch.bernoulli(adj_probs).to(self.device)  # Sample new neighbors

        return adj_logits, new_neighbors
