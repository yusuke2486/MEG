import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerAdjacency
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class AdjacencyGenerator(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, num_nodes, hidden_size, device='cpu', dropout=0.1, lr=0.001):
        super(AdjacencyGenerator, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.transformer = TransformerAdjacency(d_model, num_heads, d_ff, num_layers, num_nodes, dropout).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.to(self.device)
        print(f"AdjacencyGenerator.forward - input shape: {x.shape}")
        adj_logits = self.transformer(x)
        return adj_logits

    def generate_adjacency_logits(self, node_features):
        node_features = node_features.unsqueeze(0)  # Add batch dimension
        print(f"AdjacencyGenerator.generate_adjacency_logits - node_features shape: {node_features.shape}")
        adj_logits = self.forward(node_features)
        adj_probs = F.softmax(adj_logits, dim=-1)  # Softmax to get probabilities
        log_probs = torch.log(adj_probs + 1e-10)  # Add small value to avoid log(0)
        
        # Compute the product of all log_probs elements
        log_probs_sum = log_probs.sum()
        
        print(f"adj_logits: {adj_logits}")
        print(f"adj_probs: {adj_probs}")
        print(f"log_probs of each element: {log_probs}")
        print(f"log_probs_sum: {log_probs_sum}")

        return adj_logits.squeeze(0), log_probs_sum  # Remove batch dimension

    def update_parameters(self, log_probs, advantages):
        print("Inside update_parameters")
        print(f"log_probs: {log_probs}")
        print(f"advantages: {advantages}")

        # Calculate the policy gradient
        policy_gradient = -log_probs * advantages
        print(f"policy_gradient before sum: {policy_gradient}")

        policy_gradient = policy_gradient.sum()
        print(f"policy_gradient after sum: {policy_gradient}")

        # Ensure the policy_gradient tensor requires gradients
        policy_gradient = torch.tensor(policy_gradient, requires_grad=True)

        # Ensure the gradients are zeroed
        self.optimizer.zero_grad()
        print("optimizer.zero_grad() called")

        # Debug: check for NaNs or Infs
        if torch.isnan(policy_gradient).any():
            print("NaNs detected in policy_gradient")
        if torch.isinf(policy_gradient).any():
            print("Infs detected in policy_gradient")

        # Print out the gradients before backward
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"Before backward, {name} grad: {param.grad}")

        # Backward pass to compute gradients
        try:
            policy_gradient.backward(retain_graph=True)
            print("policy_gradient.backward() called successfully")
        except RuntimeError as e:
            print(f"Error during backward pass: {e}")
            # Debug info
            print(f"policy_gradient: {policy_gradient}")
            raise e

        # Print out the gradients after backward
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"After backward, {name} grad: {param.grad}")

        # Update parameters
        self.optimizer.step()
        print("optimizer.step() called")
