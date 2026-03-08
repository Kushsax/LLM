from passing_book_to_LLM import loss_calculator, train_data,test_data
from main_LLM import GPT
import tiktoken as tk
import torch 
import time 
start_time = time.time()

# SPEND ALL DAY TOMMORROW ONLY UNDERSTANDING HOW THIS CODE WORKS

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



def model_train(num_epochs,model,train_data,
test_data,eval_freq,start_context,optimizer):
  train_losses,test_losses = [],[]
  tokens_seen,global_step = 0,-1
  tokenizer = tk.get_encoding("gpt2")

  for epoch in range(num_epochs):
    model.train()
    for x,y in train_data:
      x=x.to(device)
      y=y.to(device)
      optimizer.zero_grad()
      logits = model(x)
      logits_flat = logits.flatten(0,1)
      targets_flat = y.flatten()
      loss = torch.nn.functional.cross_entropy(logits_flat,targets_flat)
      loss.backward() # UNDERSTAND WHAT THIS DOES
      optimizer.step() # UNDERSTAND WHAT THIS DOES
      global_step += 1
      tokens_seen += x.numel()


      if global_step % eval_freq == 0:
        model.eval()

        with torch.no_grad():
          loss1 = loss_calculator(train_data,model) 
          loss2 = loss_calculator(test_data,model)
          train_losses.append(loss1.item())
          test_losses.append(loss2.item())
          print(f"Epoch:{epoch+1} step:{global_step} train loss: {loss1.item()}test loss: {loss2.item()} ")

        model.train()
    
    with torch.no_grad():
      model.eval()
      tokens = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
      for _ in range(50):
        logits = model(tokens)
        logits = torch.argmax(logits[:,-1,:],dim=-1,keepdim=True)
        tokens = torch.cat((tokens,logits),dim=1)
      tokens = tokens.squeeze().tolist()
      text = tokenizer.decode(tokens)
      print(text.replace("\n", " "))
      model.train()
  

model = GPT(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004,weight_decay = 0.1)
num_epochs = 10
eval_freq = 5
start_context = "Every Effort Moves You"

model_train(num_epochs,model,train_data,test_data,eval_freq,start_context,optimizer)

end_time = time.time()
exec_time =  end_time - start_time
print(f"TRAINING COMPLETE EXECUTION TIME:{exec_time}")

