# TIKTOKEN DOES THIS WHOLE SCRIPT

import re 
import tiktoken as tk

with open('the-verdict.txt', 'r') as f:
  text = f.read()

tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
tokens = (items.strip() for items in tokens if items.strip())
vocab = {item:index for index,item in enumerate(tokens)}



class Tokenizer:
  def __init__(self,vocab):
    self.enc = vocab
    self.dec = {index:item for index,item in vocab.items()}
  
  def encode(self,text):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = (items.strip() for items in tokens if items.strip())
    token_ids = {self.enc[i] for i in tokens}
    return token_ids
  
  def decode(self,token_ids):
    text = " ".join([self.dec[id] for id in token_ids])
    text = re.sub(r'([,.:;?_!"()\']|--|\s)',tokens)
    return text
 
# this = Tokenizer(vocab)
# final_text = this.encode(text)
# print(len(final_text))

enc = tk.encoding_for_model("gpt2")

ids = enc.encode(text)
print(len(ids))


