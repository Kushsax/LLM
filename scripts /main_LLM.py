import torch.nn as nn
import torch 
from mha_w_weight_splits import MulteHeadAttention
torch.manual_seed(44) 
import os
torch.set_num_threads(os.cpu_count()) 
import tiktoken as tk

class LayerNormalisation(nn.Module):
  def __init__(self,token_dim):
    super().__init__()
    self.scale = nn.Parameter(torch.ones(token_dim))
    self.shift = nn.Parameter(torch.zeros(token_dim))

  def forward(self,x):
    mean = x.mean(dim=-1,keepdim=True)
    var = x.var(dim=-1,keepdim=True)
    x = (x-mean)/torch.sqrt(var + 1e-5)
    return self.scale * x + self.shift
  
class GELU(nn.Module):
  def __init__(self):
    super().__init__()
    pass
  def forward(self,x):
    return 0.5 * x * (1 + torch.tanh(
    torch.sqrt(torch.tensor(2/torch.pi)) *
    (x + 0.044715 * torch.pow(x, 3))
))
  
class FeedForward(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(cfg["token_dim"],cfg["token_dim"]*4),
      GELU(),
      nn.Linear(cfg["token_dim"]*4,cfg["token_dim"])
    )
  def forward(self,x):
    return self.layers(x)

class TransformerBlock(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.mha = MulteHeadAttention(cfg["token_dim"],cfg["token_dim"],cfg["heads"],cfg["dropout_rate"])

    self.norm1 = LayerNormalisation(cfg["token_dim"])
    self.norm2 = LayerNormalisation(cfg["token_dim"])
    self.feed_forward = FeedForward(cfg)
    self.dropout = nn.Dropout(cfg["dropout_rate"])

  def forward(self,x):
    
    shortcut = x
    x = self.norm1(x)
    x = self.mha(x)
    x = self.dropout(x)
    x = x+shortcut

    shortcut = x
    x = self.norm2(x)
    x = self.feed_forward(x)
    x = self.dropout(x)

    return x+shortcut

class GPT(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.emb = nn.Embedding(cfg["vocab_size"],cfg["token_dim"])
    self.pos_emb = nn.Embedding(cfg["context_size"],cfg["token_dim"])
    self.dropout = nn.Dropout(cfg["dropout_rate"])

    self.transformer = nn.Sequential(
    *[TransformerBlock(cfg) for _ in range(cfg["layers"])]
    )

    self.norm = LayerNormalisation(cfg["token_dim"])

    self.out_head = nn.Linear(cfg["token_dim"],cfg["vocab_size"])
    
  def forward(self,in_idx):
    batch_size, seq_len = in_idx.shape
    emb = self.emb(in_idx)
    pos = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
    x = emb+pos
    x = self.dropout(x)
    x = self.transformer(x)
    x = self.norm(x)
    logits = self.out_head(x)
    return logits

def gpt_call(model,max_new_tokens,in_idx,context_size):
  for _ in range(max_new_tokens):
    in_idx_cond = in_idx[:,-context_size:]

    logits = model(in_idx_cond)
    
    x = logits[:,-1,:]
    x = torch.softmax(x,dim=-1)
    x = torch.argmax(x,dim=-1,keepdim=True) 

    in_idx = torch.cat((in_idx,x),dim=1)
    
  return in_idx

def text_to_tokens(text,tokenizer):
  enc = tokenizer.encode(text)
  encoded_tensor = torch.tensor(enc).unsqueeze(0)
  return encoded_tensor

def tokens_to_text(tokens,tokenizer):
  dec = tokens.squeeze(0).tolist()
  dec = tokenizer.decode(dec)
  return dec

def cross_entropy_loss(inputs,targets,model):
  logits = model(inputs)
  probas = torch.softmax(logits,dim=-1)
  in_idx = 0 
  probas1 = probas[in_idx,[0,1,2],targets[in_idx]]

  in_idx = 1
  probas2 = probas[in_idx,[0,1,2],targets[in_idx]]
  target_probas = torch.cat((probas1,probas2))
  log_probas = torch.log(target_probas)
  neg_avg_log_probas = -torch.mean(log_probas)
  return neg_avg_log_probas


cfg = {
  "vocab_size": 50257,
  "token_dim": 768,
  "context_size":1024,
  "dropout_rate": 0.1,
  "heads":12,
  "layers":12
} 

inputs = torch.tensor([[16833,3626,6100],[40,1107,588]])
targets = torch.tensor([[3626,6100,345],[1107,588,11311]])

model = GPT(cfg)
res = cross_entropy_loss(inputs,targets,model)
#print(res)

logits = model(inputs)
logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()

loss = torch.nn.functional.cross_entropy(logits_flat,targets_flat)
#print(loss)





