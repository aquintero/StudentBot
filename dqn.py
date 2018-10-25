import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super(DQN, self).__init__()
        self.layers = []
        in_size = state_size
        for size in hidden_layers:
            self.layers.append(nn.Linear(in_size, size))
            in_size = size
        self.out = nn.Linear(in_size, action_size)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)
    
