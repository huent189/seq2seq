import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
    def forward(self, d_model, seq_len):
        pos = torch.arange(0, seq_len).unsqueeze(1)
        dividend = torch.pow(10000, (pos[0::2] / d_model))
        print(dividend)
        PE = torch.zeros([seq_len, d_model])
        PE.requires_grad = False
        print(pos/ dividend)
        PE[:,0::2] = torch.sin(pos/ dividend)
        PE[:,1::2] = torch.cos(pos/ dividend)
        return PE
#both source target embedding and output decoding
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
        return self.embedding(x) * np.sqrt(self.d_model)

class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
    def forward(self, Q, K, V):
        d_k = K.shape[-1]
        output = torch.matmul(Q, K.T)
        output /= np.sqrt(d_k)
        return torch.matmul(F.softmax(output, d_k),V)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiheadAttention, self).__init__()
        hid_dim = d_model // num_head
        self.Q_Linear = nn.Linear(d_model, d_model)
        self.K_Linear = nn.Linear(d_model, d_model)
        self.V_Linear = nn.Linear(d_model, d_model)
        self.num_head = num_head
    def forward(self, Q, K, V):
        q_projected = self.Q_Linear(Q)
        k_projected = self.K_Linear(K)
        v_projected = self.V_Linear(V)
        b, vc, d_model = q_projected.shape
        hid_dim = d_model // self.num_head
        q_projected = q_projected.reshape([b, self.num_head, vc, hid_dim])
        k_projected = k_projected.reshape([b, self.num_head, vc, hid_dim])
        v_projected = v_projected.reshape([b, self.num_head, vc, hid_dim])
        # attention
        atten = torch.einsum('bhqd,bhkd->bhqk', q_projected, k_projected)
        d_k = q_projected.shape[-1]
        atten /= np.sqrt(d_k)
        atten = F.softmax(atten, d_k)
        atten = torch.einsum('bhqk,bhvd')
    def attention(self, Q, K, V):
        d_k = K.shape[-1]
        output = torch.matmul(Q, K.T)
        output /= np.sqrt(d_k)
        return torch.matmul(F.softmax(output, d_k),V)

if __name__ == "__main__":
    test = Embedding(3,2)
    x = torch.LongTensor([0,1,2])
    y = test(x)
    print(y)
