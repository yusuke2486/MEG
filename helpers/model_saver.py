import torch

def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)

def save_all_weights(adj_generator, gcn_models, v_networks, final_layer):
    save_model_weights(adj_generator, 'adj_generator_weights.pth')
    for i, gcn_model in enumerate(gcn_models):
        save_model_weights(gcn_model, f'gcn_model_weights_{i}.pth')
    for i, v_network in enumerate(v_networks):
        save_model_weights(v_network, f'v_network_weights_{i}.pth')
    save_model_weights(final_layer, 'final_layer_weights.pth')
