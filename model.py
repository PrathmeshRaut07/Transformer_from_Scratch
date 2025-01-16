import torch
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
print(embedding)