import torch 

inputs = torch.tensor(
   [[0.43, 0.15, 0.89],  # Your    (x^1)
    [0.55, 0.87, 0.66],  # journey (x^2)
    [0.57, 0.85, 0.64],  # starts  (x^3)
    [0.22, 0.58, 0.33],  # with    (x^4)
    [0.77, 0.25, 0.10],  # one     (x^5)
    [0.05, 0.80, 0.55]]  # step    (x^6)
)

def logic_in_for():
  for i in range(6):
    query = inputs[i]
    attention_scores = torch.empty(inputs.shape[0])
    for i,v in enumerate(inputs):
      attention_scores[i] = torch.dot(query,v)

    attention_softmax = torch.softmax(attention_scores,dim=0)

    context_vectors = torch.empty(query.shape)
    for i,v in enumerate(inputs):
      context_vectors += attention_softmax[i]*v
    print(context_vectors)


def using_matrix():
  
  attention_scores = inputs @ inputs.T 
  attention_weights = torch.softmax(attention_scores,dim=-1)
  context_vectors = attention_weights @ inputs  

  print(context_vectors)


using_matrix()
print("_______________________________")
logic_in_for()