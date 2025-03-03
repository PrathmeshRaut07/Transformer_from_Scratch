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

class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)
    def forward(self,x):
        return self.linear_2(torch.relu(self.dropout(self.linear_1(x))))    

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


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
# import matplotlib.pyplot as plt    
# multi_head_attention = MultiHeadAttentionBlock(d_model=128, h=8, dropout=0.2)    
# # Pass the encodings through the multi-head attention block
# output = multi_head_attention(encodings, encodings, encodings,mask=None)  # Self-attention: q, k, v are the same
# print("Multi-head attention output shape:", output.shape)  # Should print torch.Size([1, 4, 128])

# # Extract attention scores
# attention_scores = multi_head_attention.attention_scores
# print("Attention scores shape:", attention_scores.shape)  # Should print torch.Size([1, 8, 4, 4])

# # Plot attention scores for the first head
# head_idx = 0  # Choose which head to visualize
# plt.figure(figsize=(8, 6))
# plt.imshow(attention_scores[0, head_idx].detach().numpy(), cmap='viridis')
# plt.colorbar()
# plt.title(f"Attention Scores (Head {head_idx + 1})")
# plt.xlabel("Key Tokens")
# plt.ylabel("Query Tokens")
# plt.show()

