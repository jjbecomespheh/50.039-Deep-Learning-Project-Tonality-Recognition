import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, dropout, bidirectional=True):
        super(LSTM, self).__init__()
        self.input_size = input_size # size of the input 
        self.hidden_size = hidden_size # number of hidden neurons
        self.layer_size = layer_size # number of layers 
        self.output_size = output_size # number of output classes
        self.bidirectional = bidirectional
        self.dropout = dropout


        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layer_size, dropout = self.dropout, batch_first = True, bidirectional = bidirectional)
        if bidirectional: self.layer = nn.Linear(hidden_size*2, output_size) # If bidirectional, we have 2 more layers
        else: self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # Set initial states
        if self.bidirectional:
            hidden_state = torch.zeros(self.layer_size*2, input.size(0), self.hidden_size) # Hidden state
            cell_state = torch.zeros(self.layer_size*2, input.size(0), self.hidden_size) # Cell state
        else:
            hidden_state = torch.zeros(self.layer_size, input.size(0), self.hidden_size) # Hidden state
            cell_state = torch.zeros(self.layer_size, input.size(0), self.hidden_size) # Cell state
            
        # LSTM:
        output, (_, _) = self.lstm(input, (hidden_state, cell_state)) # ouput, (last_hidden_state, last_cell_state) are returned
        output = output[:, -1, :]
        output = self.layer(output)        
        return output