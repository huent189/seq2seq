import torch.nn as nn
import torch
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

class SequenceEmbedding(nn.Module):
    def __init__(self):
        super(SequenceEmbedding, self).__init__()
        
    

if __name__ == "__main__":
    test = PositionalEmbedding()
    y = test(2,2)
    print(y)
