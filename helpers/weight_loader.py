import os
import torch

def load_model_weights(model, filename):
    if os.path.exists(filename):
        try:
            model.load_state_dict(torch.load(filename))
        except RuntimeError as e:
            print(f"Failed to load {filename}: {e}")
            os.remove(filename)  # Remove corrupted file
            raise FileNotFoundError(f"Corrupted file removed: '{filename}'")
    else:
        raise FileNotFoundError(f"No such file: '{filename}'")

def load_all_weights(adj_generator, gcn_models, v_networks, final_layer):
    def try_load(model, filename):
        try:
            load_model_weights(model, filename)
            print(f"Loaded {filename}")
        except FileNotFoundError as e:
            print(f"{e}")

    try_load(adj_generator, 'adj_generator_weights.pth')
    for i, gcn_model in enumerate(gcn_models):
        filename = f'gcn_model_weights_{i}.pth'
        try_load(gcn_model, filename)
    for i, v_network in enumerate(v_networks):
        filename = f'v_network_weights_{i}.pth'
        try_load(v_network, filename)
    try_load(final_layer, 'final_layer_weights.pth')
    print("Model weights loading complete. Training will continue from the available weights.")
