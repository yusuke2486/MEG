import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        # src shape: (seq_len, batch_size, d_model)
        return self.encoder(src)
