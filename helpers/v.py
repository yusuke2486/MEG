import torch
import torch.nn as nn

class VNetwork(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(VNetwork, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(d_model, 1)  # Output single value

    def forward(self, x):
        # Assuming x is of shape (batch_size, seq_len, d_model) if batch_first=True
        output = self.encoder(x)
        output = self.linear(output.mean(dim=1))  # Mean over sequence length
        return output
