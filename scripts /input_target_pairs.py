import tiktoken as tk
from torch.utils.data import Dataset, DataLoader
import torch

with open('../the-verdict.txt', 'r') as f:
  text = f.read()

class DataSet(Dataset):
  def __init__(self,text,tk,max_length,stride):
   self.input =[]
   self.target = []

   enc = tk.encoding_for_model("gpt2")
   id = enc.encode(text)

   for i in range(0,len(id) - max_length, stride):
      input_chunk = id[i:i+max_length]
      target_chunk = id[i+1:i+max_length+1]
      self.input.append(torch.tensor(input_chunk))
      self.target.append(torch.tensor(target_chunk))
  
  def __len__(self):
    return len(self.input)
  def __getitem__(self,idx):
    return self.input[idx], self.target[idx]
