import torch as t
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x