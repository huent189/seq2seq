import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, device):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.device = device

    def forward(self, x):
        pos = torch.arange(0, x.shape[1]).unsqueeze(1).float()
        dividend = torch.exp(torch.log(torch.Tensor(
            [10000])) * torch.arange(0, self.d_model, 2) / self.d_model)
        # print(dividend)
        PE = torch.zeros([x.shape[1], self.d_model]).to(self.device)
        # print(pos * dividend)
        PE[:, 0::2] = torch.sin(pos * dividend)
        PE[:, 1::2] = torch.cos(pos * dividend)
        return PE


# both source target embedding and output decoding
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.decoder.weight
        self.d_model = d_model

    def forward(self, x, decode=False):
        if decode:
            return self.decoder(x)
        x = self.embedding(x)
        return x * np.sqrt(self.d_model)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head, p_drop):
        super(MultiheadAttention, self).__init__()
        hid_dim = d_model // num_head
        self.Q_Linear = nn.Linear(d_model, d_model)
        self.K_Linear = nn.Linear(d_model, d_model)
        self.V_Linear = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)
        self.num_head = num_head
        self.qkv_droupout = nn.Dropout(p_drop)
        self.qk_droupout = nn.Dropout(p_drop)

    def forward(self, Q, K, V, mask=None):
        q_projected = self.Q_Linear(Q)
        k_projected = self.K_Linear(K)
        v_projected = self.V_Linear(V)
        b, seq_len, d_model = q_projected.shape
        hid_dim = d_model // self.num_head
        q_projected = q_projected.reshape(
            [q_projected.shape[0], q_projected.shape[1], self.num_head, hid_dim]).permute(0, 2, 1, 3)
        k_projected = k_projected.reshape(
            [k_projected.shape[0], k_projected.shape[1], self.num_head, hid_dim]).permute(0, 2, 3, 1)
        v_projected = v_projected.reshape(
            [v_projected.shape[0], v_projected.shape[1], self.num_head, hid_dim]).permute(0, 2, 1, 3)
        # attention
        atten = self.qk_droupout(torch.matmul(q_projected, k_projected))
        d_k = q_projected.shape[-1]
        atten /= np.sqrt(d_k)
        if mask is not None:
            atten.masked_fill(mask == 0, -1e10)
        atten = F.softmax(atten, dim=3)
        atten = self.qk_droupout(atten)
        atten = torch.matmul(atten, v_projected)
        # concatenate
        atten = atten.permute(0, 2, 1, 3).reshape([b, seq_len, d_model])
        atten = self.final_linear(atten)
        atten = self.qkv_droupout(atten)
        return atten


class Encoder(nn.Module):
    def __init__(self, d_model, dropout):
        """
        docstring
        """
        super(Encoder, self).__init__()
        self.attention = MultiheadAttention(d_model, 8, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2048, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2048, d_model, bias=True),
                                 nn.Dropout(dropout))

    def forward(self, x, src_mask):
        y = self.attention(x, x, x, src_mask) + x
        y = self.norm1(y)
        y = self.ffn(y) + y
        return self.norm2(y)


class Decoder(nn.Module):
    def __init__(self, d_model, p_drop):
        """
        docstring
        """
        super(Decoder, self).__init__()
        self.trg_attention = MultiheadAttention(d_model, 8, p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_attention = MultiheadAttention(d_model, 8, p_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2048, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2048, d_model, bias=True))
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tg, encoded_input, src_mask, trg_mask):
        """
        docstring
        """
        y = self.norm1(self.trg_attention(tg, tg, tg, trg_mask) + tg)
        y = self.norm2(self.encoder_attention(encoded_input, encoded_input, tg, src_mask) + y)
        return self.norm3(self.ffn(y) + y)


class Transformer(nn.Module):
    def __init__(self, vocab_size, src_pad_idx, trg_pad_idx, d_model=512, device='cuda', p_drop=0.1):
        """
        docstring
        """
        super(Transformer, self).__init__()
        self.output_token_embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEmbedding(d_model, device)
        self.src_token_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.ModuleList([Encoder(d_model, p_drop) for _ in range(6)])
        self.decoder = nn.ModuleList([Decoder(d_model, p_drop) for _ in range(6)])
        self.input_embedding_dropout = nn.Dropout(p_drop)
        self.output_embedding_dropout = nn.Dropout(p_drop)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.d_model = d_model
    
    def make_masks(self, src, trg):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_msk = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_mask = torch.tril(torch.ones((trg.shape[1], trg.shape[1]), dtype=torch.bool)).to(self.device)
        trg_mask = trg_mask & trg_pad_msk
        return src_mask, trg_mask
        
    def forward(self, x, y):
        src_mask, trg_mask = self.make_masks(x, y)
        x_token = self.src_token_embedding(x) * np.sqrt(self.d_model)
        x_pos = self.pos_encoder(x).unsqueeze(0)
        x = x_token + x_pos
        x = self.input_embedding_dropout(x)
        for layer in self.encoder:
            x = layer(x, src_mask)
        y_pos = self.pos_encoder(y).unsqueeze(0)
        y = self.output_token_embedding(y) + y_pos
        y = self.output_embedding_dropout(y)
        # decode
        for layer in self.decoder:
            y = layer(y, x, src_mask, trg_mask)
        y = self.output_token_embedding(y, decode=True)
        return F.softmax(y)

if __name__ == "__main__":
    test = Transformer(16, 1, 1, device='cpu')
    x = torch.rand([3, 16]).long()
    tg = torch.rand([3, 16]).long()
    out = test(tg, x)
