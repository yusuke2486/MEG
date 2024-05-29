import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
import time

from models.pi import AdjacencyGenerator  # 変更箇所
from models.GCN import GCN
from models.accuracy_calculator import calculate_accuracy
from helpers.data_loader import load_data, accuracy
from helpers.v import VNetwork
from helpers.sampling import sample_nodes
from helpers.weight_loader import load_all_weights
from helpers.model_saver import save_all_weights
from helpers.positional_encoding import positional_encoding

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Initialization parameters
num_nodes = 2708
num_node_features = 1433
num_classes = 7
hidden_size = 64
d_model = num_node_features
num_heads = 4
d_ff = 256
num_layers = 1
dropout = 0.1
epochs = 100
gamma = 0.99
pos_enc_dim = 7  # Number of positional encoding dimensions, adjusted to be divisible by 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

data = Data(x=features, edge_index=None, y=labels)

# Print shapes of loaded data
print(f"adj shape: {adj.shape}")
print(f"features shape: {features.shape}")
print(f"labels shape: {labels.shape}")
print(f"idx_train shape: {idx_train.shape}")
print(f"idx_val shape: {idx_val.shape}")
print(f"idx_test shape: {idx_test.shape}")

# Adjust feature size to d_model
if features.size(1) < d_model:
    features = torch.cat([features, torch.zeros(features.size(0), d_model - features.size(1))], dim=1)
elif features.size(1) > d_model:
    features = features[:, :d_model]

# Calculate positional encoding and add to features
pos_enc = positional_encoding(adj, pos_enc_dim)
features = torch.cat([features, pos_enc], dim=1)

# Print shapes after adding positional encoding
print(f"features shape after positional encoding: {features.shape}")

# num_node_combined_featuresを設定
num_node_combined_features = num_node_features + pos_enc_dim

# Move labels to device
labels = labels.to(device)
features = features.to(device)
adj = adj.to(device)  # adjもデバイスに移動

# Initialize components
adj_generator = AdjacencyGenerator(d_model + pos_enc_dim, num_heads, d_ff, num_layers, hidden_size, device, dropout).to(device)
gcn_models = [GCN(d_model + pos_enc_dim, hidden_size, num_node_combined_features).to(device) for _ in range(8)]
final_layer = torch.nn.Linear(num_node_combined_features, num_classes).to(device)  # Initialize the final layer for classification
v_networks = [VNetwork(d_model + pos_enc_dim, num_heads, d_ff, num_layers, dropout).to(device) for _ in range(8)]

# Load saved model weights if available
load_all_weights(adj_generator, gcn_models, v_networks, final_layer)

# Set up optimizers
optimizer_gcn = [optim.Adam(gcn_model.parameters(), lr=0.01) for gcn_model in gcn_models]
optimizer_v = [optim.Adam(v_network.parameters(), lr=0.01) for v_network in v_networks]
optimizer_adj = optim.Adam(adj_generator.parameters(), lr=0.01)
optimizer_final_layer = optim.Adam(final_layer.parameters(), lr=0.01)

# Create a file to log the epoch results
log_file_path = 'training_log.txt'
with open(log_file_path, 'w') as f:
    f.write("Training Log\n")

# 全てのモデルとデータを同じデバイスに移動
adj_generator.to(device)
for gcn_model in gcn_models:
    gcn_model.to(device)
for v_network in v_networks:
    v_network.to(device)
final_layer.to(device)
features = features.to(device)
adj = adj.to(device)


# Training loop
for epoch in range(epochs):
    start_time = time.time()  # Start the timer at the beginning of the epoch
    epoch_acc = 0
    epoch_mean_reward = 0

    print(f"\nEpoch {epoch + 1}/{epochs}")
    adj_generator.train()
    final_layer.train()
    for gcn_model in gcn_models:
        gcn_model.train()
    for v_network in v_networks:
        v_network.train()

    total_rewards = 0
    rewards = []
    log_probs_layers = []
    value_functions = []

    updated_features = features.clone()  # 各層で特徴量を更新

    for layer in range(8):
        print(f"\nLayer {layer + 1}/8")

        new_adj = adj.clone()  # 新しい隣接行列を初期化

        # ノードをサンプリング
        sampled_indices = sample_nodes(features, sample_ratio=0.2)  
        sampled_indices_set = set(sampled_indices.tolist())  # サンプリングされたノードのセット
        
        # For each node, generate new neighbors using Bernoulli distribution
        layer_log_probs = []
        for node_idx in range(num_nodes):
            node_feature = updated_features[node_idx].unsqueeze(0)
            neighbor_indices = adj[node_idx].nonzero().view(-1)
            neighbor_features = updated_features[neighbor_indices]

            with sdpa_kernel(SDPBackend.MATH):  # adj_generatorにmathバックエンドを適用
                adj_probs, new_neighbors = adj_generator.generate_new_neighbors(node_feature, neighbor_features)
            
            if adj_probs.isnan().any() or new_neighbors.isnan().any():
                print(f"NaN detected in adj_probs or new_neighbors at layer {layer + 1}, node {node_idx + 1}")
                continue  # スキップして次のノードへ

            if node_idx in sampled_indices_set:
                # Calculate log probabilities for the generated edges
                log_probs = new_neighbors * torch.log(adj_probs) + (1 - new_neighbors) * torch.log(1 - adj_probs)
                layer_log_probs.append(log_probs.mean())

            # Use the generated new neighbors to update the new adjacency matrix
            for i, neighbor_idx in enumerate(neighbor_indices):
                new_adj[node_idx, neighbor_idx] = new_neighbors[i]

        if layer_log_probs:
            log_probs_layers.append(torch.stack(layer_log_probs).mean())

        print(f"adj_probs: {adj_probs}")

        # Sampled nodes for computing gradient and state value function V
        sampled_features = features[sampled_indices]
        print(f"Sampled features for layer {layer + 1}: {sampled_features.shape}")

        if torch.isnan(sampled_features).any():
            print(f"NaN detected in sampled_features at layer {layer + 1} of epoch {epoch + 1}")

        # Store log probabilities and value functions for later use
        with sdpa_kernel(SDPBackend.MATH):  # VNetworkにmathバックエンドを適用
            value_function = v_networks[layer](sampled_features.unsqueeze(0)).view(-1)
        value_function = torch.clamp(value_function, min=-1000, max=1000)  # Clamp values to prevent extreme values
        value_functions.append(value_function)
        print(f"Value function for layer {layer + 1}: {value_function}")

        # Forward pass through GCN using all nodes
        edge_index, edge_weight = dense_to_sparse(new_adj)
        data.edge_index = edge_index.to(device)
        data.edge_attr = edge_weight.to(device)
        data.x = features.to(device)  # データのxをCUDAに移動
        node_features = gcn_models[layer](data.x, data.edge_index)  # featuresを明示的にデバイスに移動
        
        updated_features = node_features

        print(f"edge_index.shape: {edge_index.shape}")

        if torch.isnan(node_features).any():
            print(f"NaN detected in node_features at layer {layer + 1} of epoch {epoch + 1}")

        # Calculate reward
        sum_new_neighbors = new_adj.sum().item()  # 合計を計算
        print(f"sum_new_neighbors: {sum_new_neighbors}")
        log_sum = -torch.log(torch.tensor(sum_new_neighbors + 1, device=device))  # sum_new_neighborsをtensorに変換
        reward = log_sum.item()

        if sum_new_neighbors == 0:
            reward = 0.0

        rewards.append(reward)
        total_rewards += reward
        print(f"Reward for layer {layer + 1}: {reward}")

    # Apply final dense layer to convert to 7 classes
    output = final_layer(node_features[idx_train])
    output = F.log_softmax(output, dim=1)
    print(f'output.shape: {output.shape}')
    
    acc = accuracy(output, labels[idx_train])
    print(f"Training accuracy: {acc * 100:.2f}%")  # Print accuracy
    epoch_acc += acc
    # Calculate cumulative rewards for each layer
    cumulative_rewards = []
    for l in range(8):
        cumulative_reward = sum(rewards[l:]) + (8 * acc)
        cumulative_rewards.append(cumulative_reward)
    print(f"Cumulative rewards: {cumulative_rewards}")

    # Convert cumulative_rewards to FloatTensor
    cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float, device=device)

    # Calculate final cumulative reward with accuracy for the last layer
    final_cumulative_reward = cumulative_rewards[-1]
    print(f"Final cumulative reward with accuracy: {final_cumulative_reward}")

    # Calculate advantages
    advantages_layers = []
    for l in range(8):
        advantages = cumulative_rewards[l] - value_functions[l]
        advantages_layers.append(advantages)
        print(f"Advantages for layer {l + 1}: {advantages}")

    # Update GCN
    for opt_gcn in optimizer_gcn:
        opt_gcn.zero_grad()
    loss_gcn = F.nll_loss(output, labels[idx_train])
    print(f"GCN loss: {loss_gcn.item()}")
    loss_gcn.backward(retain_graph=True)

    # Apply gradient clipping
    for gcn_model in gcn_models:
        torch.nn.utils.clip_grad_norm_(gcn_model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(adj_generator.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(final_layer.parameters(), max_norm=1.0)
    for v_network in v_networks:
        torch.nn.utils.clip_grad_norm_(v_network.parameters(), max_norm=1.0)

    # Update Adjacency Generator
    optimizer_adj.zero_grad()

    # 各層のlog_probsの平均を計算
    log_probs_mean_layers = torch.stack(log_probs_layers)
    print(f"log_probs_mean_layers: { log_probs_mean_layers}")
    print(f"log_probs_mean_layers.shape: { log_probs_mean_layers.shape}")
    
    # lossを計算し、その勾配を使用して更新
    loss_adj = -torch.mean(log_probs_mean_layers * torch.stack(advantages_layers))  # 符号を反転
    loss_adj.backward(retain_graph=True)
    optimizer_adj.step()

    # Update V-networks
    for i, (v_network, v_opt) in enumerate(zip(v_networks, optimizer_v)):
        v_opt.zero_grad()
        v_loss = F.mse_loss(value_functions[i], torch.tensor([cumulative_rewards[i]], device=device).detach())
        print(f"V-network loss for layer {i + 1}: {v_loss.item()}")
        v_loss.backward(retain_graph=True)

    # After all gradients are computed, step the optimizers
    for opt_gcn in optimizer_gcn:
        opt_gcn.step()
    optimizer_final_layer.step()
    for v_opt in optimizer_v:
        v_opt.step()

    epoch_mean_reward = cumulative_rewards.mean()

    # Save model weights after each epoch
    save_all_weights(adj_generator, gcn_models, v_networks, final_layer)

    end_time = time.time()
    epoch_time = end_time - start_time

    # Synchronize CUDA and wait for 2 seconds to ensure all operations are complete
    torch.cuda.synchronize()
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Epoch accuracy: {epoch_acc * 100:.2f}%")  # Print average accuracy across all batches
    print(f"Epoch mean rewards: {epoch_mean_reward}")
    print(f"Advantages: {advantages_layers}")
    # Write the results to the log file
    with open(log_file_path, 'a') as f:
        f.write(f"\nEpoch {epoch + 1}/{epochs}\n")
        f.write(f"Epoch accuracy: {epoch_acc * 100:.2f}%\n")
        f.write(f"Epoch mean rewards: {epoch_mean_reward}\n")
        f.write(f"Epoch time: {epoch_time:.2f} seconds\n")  # Write the epoch time to the log file
        for i in range(8):
            f.write(f"Advantages for layer {i + 1}: {advantages_layers[i]}\n")

    time.sleep(2)

print("Training finished and model weights saved!")

# Test phase
print("Starting testing phase...")
adj_generator.eval()
final_layer.eval()
for gcn_model in gcn_models:
    gcn_model.eval()
for v_network in v_networks:
    v_network.eval()

with torch.no_grad():
    for layer in range(8):
        print(f"\nTesting Layer {layer + 1}/8")

        new_adj = adj.clone()  # 新しい隣接行列を初期化

        for node_idx in range(num_nodes):
            node_feature = features[node_idx].unsqueeze(0)
            neighbor_indices = adj[node_idx].nonzero().view(-1).to('cpu')  # Get the indices of neighbors
            neighbor_features = features[neighbor_indices].to(device)
            
            with sdpa_kernel(SDPBackend.MATH):
                adj_probs, new_neighbors = adj_generator.generate_new_neighbors(node_feature, neighbor_features)

            for i, neighbor_idx in enumerate(neighbor_indices):
                new_adj[node_idx, neighbor_idx] = new_neighbors[i]

        edge_index, edge_weight = dense_to_sparse(new_adj)
        data.edge_index = edge_index.to(device)
        data.edge_attr = edge_weight.to(device)
        data.x = features.to(device)  # テストデータもデバイスに移動
        node_features = gcn_models[layer](data.x, data.edge_index)

    output = final_layer(node_features[idx_test])
    output = F.log_softmax(output, dim=1)
    test_acc = accuracy(output, labels[idx_test])
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Write test results to the log file
    with open(log_file_path, 'a') as f:
        f.write(f"\nTest accuracy: {test_acc * 100:.2f}%\n")

print("Testing phase finished!")
