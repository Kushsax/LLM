import gensim.downloader as api
import torch
import os
from input_target_pairs import DataSet,text, tk
from torch.utils.data import  DataLoader
#READ THIS CODE BEFORE NEXT LECTURE
torch.set_num_threads(os.cpu_count()) 

# model= api.load("word2vec-google-news-300")
# vectors = model

data = DataSet(text,tk,4,4)
dataset = DataLoader(data,batch_size=8,shuffle=False,drop_last=True,num_workers=0)

#to print
it = iter(dataset)
input,target = next(it)

#vector embedding 
vocab_size = 50257
dimension = 256
emb = torch.nn.Embedding(vocab_size,dimension)

#Positional embedding
contex_length = 4
pos_embedding = torch.nn.Embedding(contex_length,dimension)
pos_emb = pos_embedding(torch.arange(contex_length))






