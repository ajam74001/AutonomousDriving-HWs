import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils

class MLP(nn.Module):
    def __init__(self, input_dim , output_dim, hidden_unit= None):
        super().__init__()
        if hidden_unit is None:
            hidden_unit = input_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_unit), nn.ReLU(), nn.Linear(hidden_unit, hidden_unit),
                                 nn.ReLU(),  nn.Linear(hidden_unit, output_dim))
    def forward(self, x):
        return self.mlp(x)

class Global_Graph(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias = False) # query
        self.k = nn.Linear(hidden_size, hidden_size, bias = False) #key
        self.v = nn.Linear(hidden_size, hidden_size, bias = False) # value
        self.p = nn.Linear(hidden_size, hidden_size) # no need 


    def forward(self, hidden_states, attention_mask=None, mapping=None):
        Q, K, V = self.q(hidden_states), self.k(hidden_states) , self.v(hidden_states)
        # scaling_factor = math.sqrt(K.size(-1)) #? 
        attention = torch.bmm(Q,K.transpose(-2, -1)) #* (1/scaling_factor) # attention definition
        if attention_mask is not None:
            attention = attention.masked_fill(attention_mask==0 , -2e9) # the neg value is correct or -inf ?
        attention = F.softmax(attention, -1)
        z= torch.bmm(attention,V)
        # proj = self.p(z)
        return z #proj


class NormMLP(nn.Module):
    def __init__(self, inp_dim, output_dim, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = inp_dim
        self.mlp = nn.Sequential(nn.Linear(inp_dim, hidden_size),
                                 nn.LayerNorm(hidden_size),
                                 nn.ReLU(), 
                                 nn.Linear(hidden_size, output_dim))
    def forward(self, x):
        return self.mlp(x)

class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()
        self.encoders = nn.ModuleList([NormMLP(hidden_size, hidden_size//2) for i in range(depth)])
        # self.maxp = nn.MaxPool1d()
        self.depth = depth # nuber of layers 
    def forward(self, hidden_states, lengths):
        # hidden_states.shape = (b, #polylines (we get the max), hidden_size(number of features(64)))
        # print(torch.tensor(lengths).shape)
        # print(max(lengths))
        for i, encoder in enumerate(self.encoders):
            x1 = encoder(hidden_states)
            # x2 , _= torch.max(x1, dim=1, keepdim  = True)# batch x 1 x hidden_size
            # x2 = self.maxp(x1.permute(0,2,1)).permute(0,2,1) # maxpooling on the 2nd dimension 
            x2 = nn.MaxPool1d(x1.shape[1], stride=1)(x1.permute(0,2,1)).permute(0,2,1)
            # print(x2.shape)
            x2 = x2.repeat(1, max(lengths), 1) # max of lenghts hidden_states.shape[1]
            hidden_states = torch.cat([x1,x2], dim=2) # batch x max#v x hidden_size
            # print(hidden_states.shape)
        # hidden_states = torch.max(hidden_states, dim=1)[0] # batch x hidden_size
        hidden_states = nn.MaxPool1d(hidden_states.shape[1], stride=1)(hidden_states.permute(0,2,1)).squeeze(-1)
        # print('hidden_states', hidden_states.shape)

        return hidden_states 
# with mask 
# class Sub_Graph(nn.Module):
#     def __init__(self, hidden_size, depth=3):
#         super(Sub_Graph, self).__init__()
#         self.hidden_size = hidden_size
#         self.layers = nn.ModuleList([NormMLP(hidden_size, hidden_size//2) for i in range(depth)])
        
#     def forward(self, hidden_states, lengths):
 
#         mask = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 1).to(x.device)
#         for i, l in enumerate(lengths):
#                 mask[i, :l, :] = 1
#         mask = mask.repeat((1, 1, self.hidden_size//2)) == 0

#         for i, layer in enumerate(self.layers):
#             x = layer(hidden_states) # #polys x max(#nodes) x hidden_size//2
#             max_x = torch.max(x.masked_fill(mask, -2e9), dim=1, keepdim=True)[0]  #polys x 1 x hidden_size//2
#             max_x = max_x.repeat(1, x.size(1), 1) # max(#nodes) x hidden_size//2
#             hidden_states = torch.cat([max_x, x], dim=-1) # #polys x max(#nodes) x hidden_size

  
#         hidden_states = torch.max(hidden_states.masked_fill(mask.repeat((1, 1, 2)), -2e9), dim=1)[0] # #polys x hidden_size

#         return hidden_states 
