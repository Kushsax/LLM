import torch 
torch.manual_seed(123)
import os
torch.set_num_threads(os.cpu_count()) 

# INPUT = 1,3,6
# Weight matrix = 6,6
# Q,K,V = 1,3,6
# with num of heads Q,K,V = 1,3,2,3
# Group by number of heads = 1,2,3,3
# attention scores = 1,2,3,3
# attention weights 
# divide by root of head dimension
# Apply softmax
# Dropout
# Context vectors 
# Transpose 1,2
# combine num_heads and head_dim 

class MulteHeadAttention(torch.nn.Module):
  def __init__(self,d_in,d_out,num_head,dropout):
    super().__init__()
    self.num_head = num_head
    self.d_in = d_in
    self.d_out = d_out 

    self.head_dim = int(d_in/num_head)

    self.Q = torch.nn.Linear(d_in,d_out)
    self.K = torch.nn.Linear(d_in,d_out)
    self.V = torch.nn.Linear(d_in,d_out)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self,x):
    b,num_tokens,d_in = x.shape
    query = self.Q(x)
    key = self.K(x)
    values = self.V(x)

    querys = query.view(b,num_tokens,self.num_head,self.head_dim)
    keys = key.view(b,num_tokens,self.num_head,self.head_dim)
    values = values.view(b,num_tokens,self.num_head,self.head_dim)

    querys = querys.transpose(1,2)
    keys = keys.transpose(1,2)
    values = values.transpose(1,2)
    

    
    attention_scores = querys @ keys.transpose(2,3)
    attention_weights = torch.tril(attention_scores)
    attention_weights = attention_weights.masked_fill(attention_weights==0,-torch.inf)
    attention_weights = torch.softmax(attention_weights/keys.shape[-1]**0.5, dim=-1)

    attention_weights = self.dropout(attention_weights)

    context_vectors = attention_weights @ values #.transpose(2,3)

    context_vectors = context_vectors.view(b,num_tokens,self.d_out)

    return context_vectors



# input = torch.tensor(torch.rand(3,6))
# batch = torch.stack((input,input), dim=0)

# print(batch)
# print("_______________________________________")

# mha = MulteHeadAttention(6,6,2)
# print(mha.forward(batch))

    
