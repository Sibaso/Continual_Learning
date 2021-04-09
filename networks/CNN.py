import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_channels=16, num_hidden=256):
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_hidden = num_hidden
        self.conv = nn.ModuleList()
        self.tasks = []

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(),
   
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=num_channels*2),
            nn.ReLU(),
    
            nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels*2, kernel_size=(3,3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.BatchNorm2d(num_features=num_channels*2),
            nn.ReLU(),
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=num_channels*2*8*8, out_features=num_hidden),
            nn.Dropout(p=0.5),
            nn.LayerNorm(num_hidden),
            nn.ReLU()
        )
        self.outputs = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)

    def add_task(self, t):
        if t in self.tasks:
            return
        self.tasks.append(t)
        self.outputs.append(
            nn.Linear(in_features=self.num_hidden, out_features=1)
        )

    def forward(self, inp):
        h = self.conv(inp)
        h = torch.flatten(h, start_dim=1)
        h = self.dense(h)
        out = []
        for head in self.outputs:
            out.append(head(h))
        out = torch.cat(out, dim=-1)
        out = self.softmax(out)
        return out
