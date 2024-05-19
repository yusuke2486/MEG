import torch

class RewardCalculator:
    def __init__(self, gamma=0.99, lambd=0.95):
        self.gamma = gamma
        self.lambd = lambd

    def reward(self, adj_matrix):
        # Example reward calculation based on the adjacency matrix
        return 1 / (adj_matrix.sum() + 1)

    def y_lambda(self, rewards, value_function):
        G_t = 0
        G_n = 0
        lambd = self.lambd
        for t in range(len(rewards)):
            # print(f"rewards[t:]: {rewards[t:].shape}")
            # print(f"value_function[t:]: {value_function[t:].shape}")
            G_t = self.G(rewards[t:], value_function[t:], self.gamma)
            G_n = G_n + (self.gamma ** t) * rewards[t]
            y_lambda_t = (1 - lambd) * G_n + lambd * G_t
        return y_lambda_t

    def G(self, rewards, value_function, gamma):
        G_t = 0
        for n in range(1, len(rewards) + 1):
            G_t += (gamma ** n) * rewards[n - 1]
        if len(value_function) > 0:
            G_t += (gamma ** len(rewards)) * value_function[min(len(rewards) - 1, len(value_function) - 1)]
        return G_t

def save_rewards(rewards, filepath):
    torch.save(rewards, filepath)
