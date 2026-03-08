import torch 
import tiktoken as tk 
from input_target_pairs import DataSet,DataLoader,text
from main_LLM import GPT,cross_entropy_loss
torch.manual_seed(44) 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

cfg = {
  "vocab_size": 50257,
  "token_dim": 768,
  "context_size":256,
  "dropout_rate": 0.1,
  "heads":12,
  "layers":12
} 
train_ratio = 0.9
split = int(train_ratio * len(text))
train_text = text[:split]
test_text = text[split:]

train_data = DataSet(train_text,tk,cfg["context_size"],cfg["context_size"]) 
train_data = DataLoader(train_data,2,False,drop_last=False,num_workers=0)


test_data = DataSet(test_text,tk,cfg["context_size"],cfg["context_size"]) 
test_data = DataLoader(test_data,2,False,drop_last=False,num_workers=0)

def loss_calculator(data, model, num_batches=10):

  total_loss = 0
  device = next(model.parameters()).device

  for i, (x,y) in enumerate(data):

    if num_batches is not None and i >= num_batches:
        break

    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    logits_flat = logits.flatten(0,1)
    targets_flat = y.flatten()

    loss = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat
    )

    total_loss += loss

  return total_loss / (i+1)

model = GPT(cfg)
loss_train = loss_calculator(train_data,model)
loss_test = loss_calculator(test_data,model)
#print(loss_train,loss_test) 

#UNDERSTAND HOW TO USE CUDA,CPU AND M3



