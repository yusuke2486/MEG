import torch

def sample_nodes(features, sample_ratio=0.2):
    num_nodes = features.size(0)
    sample_size = int(num_nodes * sample_ratio)
    sampled_indices = torch.randperm(num_nodes)[:sample_size]
    return sampled_indices
