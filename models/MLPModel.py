import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size # size of the input 
        self.hidden_size_1 = hidden_size_1 # number of hidden neurons 1
        self.hidden_size_2 = hidden_size_2 # number of hidden neurons 2
        self.output_size = output_size # number of output classes

        self.input_fc = nn.Linear(input_size, hidden_size_1)
        self.hidden_fc = nn.Linear(hidden_size_1, hidden_size_2)
        self.output_fc = nn.Linear(hidden_size_2, output_size)

    def forward(self, input):
        batch_size = input.shape[0] # (batch_size, height, width)
        input = input.view(batch_size, -1) # (batch_size, height * width)
        hidden_1 = F.relu(self.input_fc(input))
        hidden_2 = F.relu(self.hidden_fc(hidden_1))
        y_pred = self.output_fc(hidden_2)
        return y_pred