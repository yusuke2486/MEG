import torch

def calculate_cumulative_reward(rewards, accuracy, gamma):
    cumulative_reward = 0
    for reward in rewards:
        cumulative_reward += reward / sum(torch.exp(reward))
    cumulative_reward += len(rewards) * accuracy
    return cumulative_reward
