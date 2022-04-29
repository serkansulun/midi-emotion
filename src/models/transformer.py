import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

class Transformer(nn.Module):

    def __init__(self, n_tokens=None, n_layer=None, n_head=None, d_model=None, d_ff=None,
                dropout=0.0, pad_idx=0):
        super(Transformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # self.name = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ff, dropout=dropout)
        norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer, norm=norm)
        self.encoder = nn.Embedding(n_tokens, d_model, padding_idx=pad_idx)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_tokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_key_padding_mask=None):
        
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask,
                    src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)