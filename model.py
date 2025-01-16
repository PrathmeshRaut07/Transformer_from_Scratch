import torch
import math
import torch.nn as nn
class InputEmbedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x)   
src_model=InputEmbedding(27,128)
token=torch.tensor([0,1,2,3])
#0=a
#1=b
embedding=src_model(token)
print(embedding.shape)

class PositionalEmbedding(nn.Module):
    def __init__(self,seq_len:int,d_model,dropout_rate:float):
        super().__init__()
        self.seq_len=seq_len
        self.d_model=d_model
        self.dropout_rate=dropout_rate
        self.dropout=nn.Dropout(self.dropout_rate)
        pe=torch.zeros(seq_len,d_model)
        position=torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position * div_term) 
        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,x):
        x=x+(self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

new=PositionalEmbedding(4,128,0.2)
encodings=new(embedding)
print(encodings)


