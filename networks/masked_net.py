import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.bayesian_layer import BayesianLinear
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Model(nn.Module):
    def __init__(self, num_layers, num_hidden=256):
        super(Model, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.tasks = []
        self.masks = []

        self.inputs = BayesianLinear(in_features=28*28, out_features=num_hidden)

        self.relu = nn.ReLU(inplace =False)
           
        self.hiddens = nn.ModuleList([
            BayesianLinear(in_features=num_hidden, out_features=num_hidden)  
        for _ in range(num_layers)])
        self.norm = nn.LayerNorm(num_hidden)
        self.outputs = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)

    def add_task(self, t):
        if t in self.tasks:
            return
        self.tasks.append(t)
        self.outputs.append(
            BayesianLinear(in_features=self.num_hidden, out_features=1)
        )


    def forward(self, inp, sample=False):
        inp = torch.flatten(inp, start_dim=1)
        h = self.relu(self.inputs(inp, sample))

        for i in range(self.num_layers):
            h = self.relu(self.hiddens[i](h, sample))
            
        out = []
        for head in self.outputs:
            out.append(head(h, sample))
        out = torch.cat(out, dim=-1)
        out = self.softmax(out)
        return out
