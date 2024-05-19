import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from models.adj_generator import AdjacencyGenerator
from models.environment import GCN
from helpers.data_loader import load_data
from helpers.q import QNetwork
from helpers.v import VNetwork
from helpers.y import RewardCalculator, save_rewards

# モデルの重みを保存する関数
def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)

# モデルの重みを読み込む関数
def load_model_weights(model, filename):
    model.load_state_dict(torch.load(filename))

# 初期化パラメータ
num_nodes = 2708
num_node_features = 1440
num_classes = 7
hidden_size = 64
d_model = num_node_features
num_heads = 8
d_ff = 256
num_layers = 4
dropout = 0.1
epochs = 100
gamma = 0.99
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# データの読み込み
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# 特徴量のサイズをd_modelに調整
if features.size(1) < d_model:
    features = torch.cat([features, torch.zeros(features.size(0), d_model - features.size(1))], dim=1)
elif features.size(1) > d_model:
    features = features[:, :d_model]

# GCN互換のデータオブジェクトを作成
data = Data(x=features, edge_index=None, y=labels)

# コンポーネントの初期化
adj_generator = AdjacencyGenerator(d_model, num_heads, d_ff, num_layers, num_nodes, hidden_size, device, dropout).to(device)
gcn_model = GCN(num_node_features, num_node_features, hidden_size).to(device)
q_network = QNetwork(num_node_features, num_nodes, hidden_size).to(device)
v_network = VNetwork(num_node_features, hidden_size).to(device)
reward_calculator = RewardCalculator(gamma=gamma)

# 保存されたモデルの重みを読み込む（必要な場合）
try:
    load_model_weights(adj_generator, 'adj_generator_weights.pth')
    load_model_weights(gcn_model, 'gcn_model_weights.pth')
    load_model_weights(q_network, 'q_network_weights.pth')
    load_model_weights(v_network, 'v_network_weights.pth')
    print("Successfully loaded saved model weights.")
except FileNotFoundError:
    print("No saved model weights found. Training from scratch.")

# オプティマイザの設定
optimizer_gcn = optim.Adam(gcn_model.parameters(), lr=0.001)
optimizer_q = optim.Adam(q_network.parameters(), lr=0.001)
optimizer_v = optim.Adam(v_network.parameters(), lr=0.001)
optimizer_adj = adj_generator.optimizer

# トレーニングループ
for epoch in range(epochs):
    adj_generator.train()
    gcn_model.train()
    q_network.train()
    v_network.train()

    node_features = features
    total_rewards = 0
    for layer in range(10):
        # Generate adjacency matrix
        adj_logits, log_probs = adj_generator.generate_adjacency_logits(node_features)
        print(f"Generated adjacency logits for layer {layer + 1}: {adj_logits.shape}")
        print(f"Log probabilities: {log_probs}")

        adj_probs = torch.softmax(adj_logits, dim=-1)
        print(f"Adjacency probabilities: {adj_probs.shape}")

        # Convert adjacency matrix to edge_index and edge_weight
        edge_index, edge_weight = dense_to_sparse(adj_probs.squeeze(0))
        print(f"Edge index: {edge_index.shape}, Edge weight: {edge_weight.shape}")

        # Update data object
        data.edge_index = edge_index
        data.edge_attr = edge_weight

        # Forward pass through GCN
        node_features = gcn_model(data.x, data.edge_index)
        print(f"GCN output for layer {layer + 1}: {node_features.shape}")

        # 各層での報酬計算
        reward = reward_calculator.reward(adj_probs)
        total_rewards += reward

        # 各層でのV値の計算
        value_function = v_network(features.unsqueeze(0)).view(-1)
        print(f"Value function shape: {value_function.shape}, Value function: {value_function}")

        # 各層でのQ値の計算
        q_values = q_network(features.unsqueeze(0), adj_probs.unsqueeze(0)).view(-1)
        print(f"Q values shape: {q_values.shape}, Q values: {q_values}")

        # 各層でのAdvantageの計算
        advantages = q_values - value_function
        print(f"Advantages shape: {advantages.shape}, Advantages: {advantages}")

        # 各層でのY λ値の計算
        y_lambda_values = reward_calculator.y_lambda(torch.tensor([reward for _ in range(len(labels))], device=device), value_function).view(-1)
        print(f"Y lambda values shape: {y_lambda_values.shape}, Y lambda values: {y_lambda_values}")

        # Ensure shapes match for loss calculation
        min_size = min(q_values.size(0), y_lambda_values.size(0))
        q_values = q_values[:min_size]
        y_lambda_values = y_lambda_values[:min_size]
        value_function = value_function[:min_size]
        advantages = advantages[:min_size]

        print(f"Trimmed Q values shape: {q_values.shape}, Trimmed Y lambda values shape: {y_lambda_values.shape}")
        print(f"Trimmed Value function shape: {value_function.shape}")
        print(f"Trimmed Advantages shape: {advantages.shape}")

        # Update Q-network
        optimizer_q.zero_grad()
        q_loss = F.mse_loss(q_values, y_lambda_values)
        print(f"Q loss: {q_loss.item()}")
        q_loss.backward(retain_graph=True)
        optimizer_q.step()
        print("Q is updated!")

        # Update V-network
        optimizer_v.zero_grad()
        v_loss = F.mse_loss(value_function, y_lambda_values)
        print(f"V loss: {v_loss.item()}")
        v_loss.backward(retain_graph=True)
        optimizer_v.step()
        print("V is updated!")

        # Update Adjacency Generator
        adj_generator.update_parameters(log_probs, advantages)

    # 最後のノード特徴量を2クラス分類に使用
    output = node_features
    loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
    print(f"GCN loss: {loss_gcn.item()}")

    # 最後に正解率に基づいて報酬を加算
    accuracy_reward = ...  # 正解率に基づいて計算するロジックを追加
    total_rewards += accuracy_reward

    # トレーニング終了後にモデルの重みを保存
    save_model_weights(adj_generator, 'adj_generator_weights.pth')
    save_model_weights(gcn_model, 'gcn_model_weights.pth')
    save_model_weights(q_network, 'q_network_weights.pth')
    save_model_weights(v_network, 'v_network_weights.pth')

print("Training finished and model weights saved!")
