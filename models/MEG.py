import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from adj_generator import AdjacencyGenerator
from environment import GCN

class MEG:
    def __init__(self, num_node_features, num_classes, hidden_size, d_model, num_heads, d_ff, num_layers, dropout, device='cpu'):
        self.device = device

        # Initialize Adjacency Generator
        self.adj_generator = AdjacencyGenerator(d_model, num_heads, d_ff, num_layers, num_node_features, device, dropout).to(device)

        # Initialize GCN
        self.gcn = GCN(num_node_features, num_classes, hidden_size).to(device)

        # Optimizers
        self.optimizer_adj = optim.Adam(self.adj_generator.parameters(), lr=0.001)
        self.optimizer_gcn = optim.Adam(self.gcn.parameters(), lr=0.001)

    def train_step(self, node_features, labels):
        self.adj_generator.train()
        self.gcn.train()

        # Generate adjacency matrix
        adj_matrix = self.adj_generator.generate_adjacency_matrix(node_features)

        # Convert adjacency matrix to edge_index and edge_weight
        edge_index, edge_weight = dense_to_sparse(adj_matrix)

        # Create data object for GCN compatibility
        data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)

        # Forward pass through GCN
        output = self.gcn(data, adj_matrix)
        loss = F.nll_loss(output, labels)

        # Backpropagation
        self.optimizer_gcn.zero_grad()
        loss.backward()
        self.optimizer_gcn.step()

        return loss.item()

    def train(self, node_features, labels, epochs=100):
        for epoch in range(epochs):
            loss = self.train_step(node_features, labels)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

    def evaluate(self, node_features, labels):
        self.adj_generator.eval()
        self.gcn.eval()

        with torch.no_grad():
            adj_matrix = self.adj_generator.generate_adjacency_matrix(node_features)
            edge_index, edge_weight = dense_to_sparse(adj_matrix)
            data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
            output = self.gcn(data, adj_matrix)
            loss = F.nll_loss(output, labels)
            pred = output.argmax(dim=1)
            accuracy = pred.eq(labels).sum().item() / labels.size(0)

        print(f'Evaluation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
        return loss.item(), accuracy

if __name__ == "__main__":
    # Example parameters
    num_nodes = 34
    num_node_features = 128
    num_classes = 7
    hidden_size = 64
    d_model = num_node_features
    num_heads = 8
    d_ff = 256
    num_layers = 4
    dropout = 0.1
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Example data
    node_features = torch.rand(num_nodes, num_node_features).to(device)
    labels = torch.randint(0, num_classes, (num_nodes,)).to(device)

    # Initialize MEG model
    meg_model = MEG(num_node_features, num_classes, hidden_size, d_model, num_heads, d_ff, num_layers, dropout, device)

    # Train the model
    meg_model.train(node_features, labels, epochs)

    # Evaluate the model
    meg_model.evaluate(node_features, labels)
