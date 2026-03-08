from sa_w_trainable_weights import Attention
import torch

# LEARN THIS BETTER BEFORE MOVING ON (ASK QUESTIONS )

inputs = torch.tensor(
   [[0.43, 0.15, 0.89],  # Your    (x^1)
    [0.55, 0.87, 0.66],  # journey (x^2)
    [0.57, 0.85, 0.64],  # starts  (x^3)
    [0.22, 0.58, 0.33],  # with    (x^4)
    [0.77, 0.25, 0.10],  # one     (x^5)
    [0.05, 0.80, 0.55]]  # step    (x^6)
) 
batch = torch.stack((inputs,inputs), dim = 0)

class multi_head_attention(torch.nn.Module):
  def __init__(self,num_heads, d_in,d_out):
    super().__init__()
    self.heads = torch.nn.ModuleList(Attention(d_in,d_out)for i in range(num_heads))
  
  def forward(self,x):
    return torch.cat([head(x) for head in self.heads], dim =-1)

mha = multi_head_attention(2,3,2)
cv = mha.forward(batch)
print(cv)