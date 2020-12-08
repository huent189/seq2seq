import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self):
        pos = torch.arange(0, self.seq_len).unsqueeze(1).float()
        dividend = 1 / torch.pow(10000, (torch.arange(0, self.d_model, 2) / self.d_model)).float()
        print(dividend)
        PE = torch.zeros([self.seq_len, self.d_model])
        PE.requires_grad = False
        print(pos * dividend)
        PE[:, 0::2] = torch.sin(pos * dividend)
        PE[:, 1::2] = torch.cos(pos * dividend)
        return PE
# both source target embedding and output decoding


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.decoder.weight
        self.d_model = d_model

    def forward(self, x, decode=False):
        if decode:
            return self.decoder(x)
        x = self.embedding(x)
        print('eb', x.shape)
        return x * np.sqrt(self.d_model)


class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()

    def forward(self, Q, K, V):
        d_k = K.shape[-1]
        output = torch.matmul(Q, K.T)
        output /= np.sqrt(d_k)
        return torch.matmul(F.softmax(output, d_k), V)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head, masked=False):
        super(MultiheadAttention, self).__init__()
        hid_dim = d_model // num_head
        self.Q_Linear = nn.Linear(d_model, d_model)
        self.K_Linear = nn.Linear(d_model, d_model)
        self.V_Linear = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)
        self.num_head = num_head
        self.masked = masked

    def forward(self, Q, K, V):
        q_projected = self.Q_Linear(Q)
        k_projected = self.K_Linear(K)
        v_projected = self.V_Linear(V)
        print(q_projected.shape)
        b, vc, d_model = q_projected.shape
        hid_dim = d_model // self.num_head
        q_projected = q_projected.reshape([b, vc, self.num_head, hid_dim])
        k_projected = k_projected.reshape([b, vc, self.num_head, hid_dim])
        v_projected = v_projected.reshape([b, vc, self.num_head, hid_dim])
        # attention
        atten = torch.einsum('bqhd,bkhd->bhqk', q_projected, k_projected)
        d_k = q_projected.shape[-1]
        atten /= np.sqrt(d_k)
        if self.masked:
            mask = torch.triu(torch.ones(atten.shape, dtype=torch.bool), diagonal=1)
            atten[mask] = -float('inf')
        atten = F.softmax(atten, 3)
        atten = torch.einsum('bhqd,bdhv->bqhv', atten, v_projected)
        # concatenate
        atten = atten.reshape([b, vc, d_model])
        atten = self.final_linear(atten)
        return atten


class Encoder(nn.Module):
    def __init__(self, d_model):
        """
        docstring
        """
        super(Encoder, self).__init__()
        self.attention = MultiheadAttention(d_model, 8)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2048, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2048, d_model, bias=True))

    def forward(self, x):
        y = self.attention(x, x, x) + x
        y = self.norm1(y)
        y = self.ffn(y) + y
        return self.norm2(y)

class Decoder(nn.Module):
    def __init__(self, d_model):
        """
        docstring
        """
        super(Decoder, self).__init__()
        self.masked_attention = MultiheadAttention(d_model, 8)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiheadAttention(d_model, 8)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2048, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2048, d_model, bias=True))
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tg, encoded_input):
        """
        docstring
        """
        y = self.norm1(self.masked_attention(tg, tg, tg) + tg)
        y = self.norm2(self.attention(encoded_input, encoded_input, tg) + y)
        return self.norm3(self.ffn(y) + y)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model= 512):
        """
        docstring
        """
        super(Transformer, self).__init__()
        self.embbeder = Embedding(vocab_size, d_model)
        pos_encoder = PositionalEmbedding(d_model, d_model)
        self.encoded_pos = pos_encoder()
        self.encoder = Encoder(d_model)
        self.decoder = Decoder(d_model)

    def forward(self, x, y):
        embed_x = self.embbeder(x)
        print(embed_x.shape)
        x = embed_x + self.encoded_pos
        print(x.shape)
        x = self.encoder(x)
        y = self.embbeder(y) + self.encoded_pos
        y = self.decoder(y, x)
        return F.softmax(y)
if __name__ == "__main__":
    test = Transformer(16, 8)
    x = torch.rand([3, 16]).long()
    tg = torch.rand([3, 16]).long()
    out = test(tg, x)
    print(out.shape)
