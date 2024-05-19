import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, num_node_features, num_nodes, hidden_size):
        super(QNetwork, self).__init__()
        input_dim = num_node_features * num_nodes + num_nodes * num_nodes
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output is a single Q-value

    def forward(self, state, action):
        batch_size = state.size(0)
        action_flat = action.view(batch_size, -1)  # (batch_size, num_nodes * num_nodes)
        state_flat = state.view(batch_size, -1)  # (batch_size, num_node_features * num_nodes)
        
        print(f"state_flat shape: {state_flat.shape}")
        print(f"action_flat shape: {action_flat.shape}")
        
        x = torch.cat([state_flat, action_flat], dim=1)  # (batch_size, num_node_features * num_nodes + num_nodes * num_nodes)
        
        print(f"Concatenated input shape: {x.shape}")
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
