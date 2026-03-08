import torch

class Attention(torch.nn.Module):

  def __init__(self,d_in,d_out):
    super().__init__()
    self.W_query = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
    self.W_key = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
    self.W_value = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
    self.dropout = torch.nn.Dropout(0.5)

  def forward(self,x): 
    query_matrix = x @ self.W_query
    key_matrix = x @ self.W_key
    value_matrix = x @ self.W_value
    
    attension_scores = query_matrix @ key_matrix.transpose(1,2)
    #Transpose kyu kar rahe hai? - becayse dot prodcut hai 

    masked_attention_weights = torch.tril(attension_scores)

    masked_attention_weights_inf = masked_attention_weights.masked_fill(masked_attention_weights==0,-torch.inf) 

    attention_weights = torch.softmax(masked_attention_weights_inf/2**0.5,dim=-1)

    context_vectors = attention_weights @ value_matrix 
    return context_vectors
  
attention = Attention(d_in=3, d_out=2)
x = torch.rand(2, 6, 3)
output = attention.forward(x)
 
# print("Input shape:", x.shape)
# print("Output shape:", output.shape)
# print(output)




# inputs = torch.tensor(
#    [[0.43, 0.15, 0.89],  # Your    (x^1)
#     [0.55, 0.87, 0.66],  # journey (x^2)
#     [0.57, 0.85, 0.64],  # starts  (x^3)
#     [0.22, 0.58, 0.33],  # with    (x^4)
#     [0.77, 0.25, 0.10],  # one     (x^5)
#     [0.05, 0.80, 0.55]]  # step    (x^6)
# ) 


# W_query = torch.nn.Parameter(torch.rand(3,2),requires_grad=False)
# W_key = torch.nn.Parameter(torch.rand(3,2),requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(3,2),requires_grad=False)

# #More commam ot use nn.linear to initialise the matrices  

# query_matrix = inputs @ W_query
# key_matrix = inputs @ W_key
# value_matrix = inputs @ W_value

# attension_scores = query_matrix @ key_matrix.T

# # NEURONS IN A NEURAL NETWORK GET LAZY SO DROPOUT IS USED WHERE RANDOMLY SOME NODES ARE SWITCHED OFF

# dropout = torch.nn.Dropout(0.5)

# #CAUSAL SELF ATTENTION(DONT SHOW WORDS AFTER ONLY BEFORE)

# masked_attention_weights = torch.tril(attension_scores)

# #SO YOU CANT SOFTMAX THE ATTENTION SCORES BEFORE UPPER TRIANGLE SO WE DO IT AFTER, FOR THAT THE 0 BECOME -INF BECAUSE SOFTMAT USES E^z SO THE IT BECOMES 0

# masked_attention_weights_inf = masked_attention_weights.masked_fill(masked_attention_weights==0,-torch.inf)

# attention_weights = torch.softmax(masked_attention_weights_inf/2**0.5,dim=-1)

# attention_weights = dropout(attention_weights)


# context_vectors = attention_weights @ value_matrix 

# print(context_vectors)


