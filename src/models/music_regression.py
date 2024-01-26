import torch
import math as m
import numpy as np
import math
import torch.nn.functional as F
import sys

# from torch.nn.modules.activation import ReLU

sys.path.append("..")
# from utils import memory


"""
MUSIC TRANSFORMER REGRESSION (to output emotion)
"""

def generate_mask(x, pad_token=None, batch_first=True):

    batch_size = x.size(0)
    seq_len = x.size(1)

    subsequent_mask = torch.logical_not(torch.triu(torch.ones(seq_len, seq_len, device=x.device)).t()).unsqueeze(
        -1).repeat(1, 1, batch_size)
    pad_mask = x == pad_token
    if batch_first:
        pad_mask = pad_mask.t()
    mask = torch.logical_or(subsequent_mask, pad_mask)
    if batch_first:
        mask = mask.permute(2, 0, 1)
    return mask


class MusicRegression(torch.nn.Module):
    def __init__(self, embedding_dim=None, d_inner=None, vocab_size=None, num_layer=None, num_head=None,
                 max_seq=None, dropout=None, pad_token=None, output_size=None, 
                 d_condition=-1, no_mask=True, attn_type=None,
                 ):
        super().__init__()

        # assert d_condition <= 0

        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.pad_token = pad_token

        self.no_mask = no_mask

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, 
                                            embedding_dim=self.embedding_dim,
                                            padding_idx=pad_token)


        self.pos_encoding = DynamicPositionEmbedding(self.embedding_dim, max_seq=max_seq)

        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(embedding_dim, d_inner, dropout, h=num_head, additional=False, max_seq=max_seq)
             for _ in range(num_layer)])
        self.dropout = torch.nn.Dropout(dropout)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, output_size),
            torch.nn.Tanh()
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):

        mask = None if self.no_mask else generate_mask(x, self.pad_token)
        # embed input
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= math.sqrt(self.embedding_dim)

        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x, mask)

        x = self.fc(x[:, 0, :])

        return x

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_inner, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, d_inner)
        self.FFN_suf = torch.nn.Linear(d_inner, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):
        attn_out = self.rga([x,x,x], mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = F.relu(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2
    
def sinusoid(max_seq, embedding_dim):
    return np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])


class DynamicPositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        embed_sinusoid_list = sinusoid(max_seq, embedding_dim)

        self.positional_embedding = torch.from_numpy(embed_sinusoid_list).to(
            self.device, dtype=self.dtype)

    def forward(self, x):
        if x.device != self.device or x.dtype != self.dtype:
            self.positional_embedding = self.positional_embedding.to(x.device, dtype=x.dtype)
        x += self.positional_embedding[:, :x.size(1), :]
        return x


class RelativeGlobalAttention(torch.nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb
        self.E = torch.nn.Parameter(torch.randn([self.max_seq, int(self.dh)]))
        if self.additional:
            self.Radd = None

    def forward(self, inputs, mask=None):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        v = inputs[2]
        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.size(2)
        self.len_q = q.size(2)

        E = self._get_left_embedding(self.len_q, self.len_k).to(q.device)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            mask = mask.unsqueeze(1)
            new_mask = torch.zeros_like(mask, dtype=torch.float)
            new_mask.masked_fill_(mask, float("-inf"))
            mask = new_mask
            logits += mask

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), -1, self.d))

        out = self.fc(out)
        return out

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe

def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)