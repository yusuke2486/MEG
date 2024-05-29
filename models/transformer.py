import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory):
        output = self.decoder(tgt, memory)
        return output
