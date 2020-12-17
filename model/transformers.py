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


# # both source target embedding and output decoding
# class Embedding(nn.Module):
#     def __init__(self, vocab_size, d_model):
#         super(Embedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.decoder = nn.Linear(d_model, vocab_size, bias=False)
#         self.embedding.weight = self.decoder.weight
#         self.d_model = d_model

#     def forward(self, x, decode=False):
#         if decode:
#             return self.decoder(x)
#         x = self.embedding(x)
#         return x * np.sqrt(self.d_model)


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
            atten = atten.masked_fill(mask == 0, -1e10)
        atten = F.softmax(atten, dim=3)
        atten = self.qk_droupout(atten)
        atten = torch.matmul(atten, v_projected)
        # concatenate
        atten = atten.permute(0, 2, 1, 3).reshape([b, seq_len, d_model])
        atten = self.final_linear(atten)
        atten = self.qkv_droupout(atten)
        return atten


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout):
        """
        docstring
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, 8, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2048, bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(2048, d_model, bias=True))

    def forward(self, x, src_mask):
        y = self.attention(x, x, x, src_mask) + x
        y = self.norm1(y)
        y = self.ffn(y) + y
        return self.norm2(y)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, p_drop):
        """
        docstring
        """
        super(DecoderLayer, self).__init__()
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
        y = self.norm2(self.encoder_attention(y, encoded_input, encoded_input, src_mask) + y)
        return self.norm3(self.ffn(y) + y)

class Encoder(nn.Module):
    def __init__(self, d_model, p_drop, src_vocab_size, n_layers, device):
        super(Encoder, self).__init__()
        self.tok_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(d_model, device)
        self.d_model = d_model
        self.do = nn.Dropout(p_drop)
        self.encode_layers = nn.ModuleList([EncoderLayer(d_model, p_drop) for _ in range(n_layers)])
    def forward(self, x, src_mask):
        # x.shape: b, seq_len
        seq_len = x.shape[1]
        encoded_tok = self.tok_embedding(x) * (self.d_model ** (-0.5))
        encoded_pos = self.pos_embedding(x)
        encoded_x = self.do(encoded_pos + encoded_tok)
        for layer in self.encode_layers:
            encoded_x = layer(encoded_x, src_mask)
        return encoded_x

class Decoder(nn.Module):
    def __init__(self, d_model, p_drop, trg_vocab_size, n_layers, device):
        super(Decoder, self).__init__()
        self.tok_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(trg_vocab_size, d_model)
        self.d_model = d_model
        self.do = nn.Dropout(p_drop)
        self.decode_layers = nn.ModuleList([DecoderLayer(d_model, p_drop) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, trg_vocab_size)
    def forward(self, x, y, src_mask, trg_mask):
        encoded_tok = self.tok_embedding(y) * (self.d_model ** (-0.5))
        encoded_pos = self.pos_embedding(y)
        encoded_y = self.do(encoded_tok + encoded_pos)
        for layer  in self.decode_layers:
            encoded_y = layer(encoded_y, x, src_mask, trg_mask)
        output = self.fc(encoded_y)
        return output
        
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, n_layers = 6, d_model=512, device='cuda', p_drop=0.1):
        """
        docstring
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, p_drop, src_vocab_size, n_layers, device)
        self.decoder = Decoder(d_model, p_drop, trg_vocab_size, n_layers, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_masks(self, src, trg):
        #mask shape: b, 1, seq_len, seq_len
        #seq shape: b, shape
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_msk = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_mask = torch.tril(torch.ones((trg.shape[1], trg.shape[1]), dtype=torch.bool)).to(self.device)
        trg_mask = trg_mask & trg_pad_msk
        return src_mask, trg_mask
        
    def forward(self, x, y):
        src_mask, trg_mask = self.make_masks(x, y)
        context = self.encoder(x, src_mask)
        output = self.decoder(context, y, src_mask, trg_mask)
        return output
    def translate_sentence(self, x, src_vocab, trg_vocab, max_len=200):
        # x.shape: seq_len
        eval()
        trg_input = [trg_vocab.vocab.stoi[trg_vocab.init_token]] * max_len
        trg_input = torch.LongTensor(trg_input).unsqueeze(0).to(self.device)
        src_mask, trg_mask = self.make_masks(x, trg_input)
        encoded_x = self.encoder(x, src_mask)
        last_idx = -1
        for i in range(max_len):
            pred = self.decoder(encoded_x, trg_input, src_mask, trg_mask)
            pred = pred.argmax(dim=-1)
            trg_input[0,i+1] = pred[0,i]
            if pred[0,i] == trg_vocab.vocab.stoi[trg_vocab.eos_token]:
                last_idx = i
                break
        final_pred = trg_input[0,:i]
        trg_tokens = [trg_vocab.vocab.itos[i] for i in final_pred]
        print(trg_tokens)
        return final_pred

# if __name__ == "__main__":
#     # test = Transformer(16, 1, 1, device='cpu')
#     # x = torch.rand([3, 16]).long()
#     # tg = torch.rand([3, 16]).long()
#     # out = test(tg, x)
