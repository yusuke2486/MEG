import torch
import torch.nn as nn
import torch.nn.functional as F

class VNetwork(nn.Module):
    def __init__(self, num_node_features, hidden_size):
        super(VNetwork, self).__init__()
        self.fc1 = nn.Linear(num_node_features * 2708, hidden_size)  # Adjusting the input size
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output is a single V-value

    def forward(self, state):
        x = state.view(state.size(0), -1)  # Flatten the node features
        print(f"Input shape to fc1: {x.shape}")
        x = F.relu(self.fc1(x))
        print(f"Shape after fc1: {x.shape}")
        x = F.relu(self.fc2(x))
        print(f"Shape after fc2: {x.shape}")
        v_value = self.fc3(x)
        print(f"Shape after fc3: {v_value.shape}")
        return v_value  # Returning the output without squeezing it
