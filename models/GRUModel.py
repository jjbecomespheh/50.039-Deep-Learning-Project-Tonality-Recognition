import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, bidirectional=True, dropout=0.2):
        super(GRU, self).__init__()
        self.input_size = input_size  # size of the input
        self.hidden_size = hidden_size  # number of hidden neurons
        self.layer_size = layer_size  # number of layers
        self.output_size = output_size  # number of output classes
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(
            self.input_size, 
            self.hidden_size, 
            self.layer_size, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout = self.dropout
        )
        if bidirectional:
            # If bidirectional, we have 2 more layers
            self.layer = nn.Linear(hidden_size*2, output_size)
        else:
            self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # Set initial states
        if self.bidirectional:
            hidden_state = torch.zeros(
                self.layer_size*2, input.size(0), self.hidden_size)  # Hidden state
            cell_state = torch.zeros(
                self.layer_size*2, input.size(0), self.hidden_size)  # Cell state
        else:
            hidden_state = torch.zeros(self.layer_size, input.size(0), self.hidden_size)  # Hidden state
            cell_state = torch.zeros(self.layer_size, input.size(0), self.hidden_size)  # Cell state

        # LSTM:
        # ouput, (last_hidden_state, last_cell_state) are returned
        output, (_, _) = self.rnn(input, (hidden_state, cell_state))
        output = output[:, -1, :]
        output = self.layer(output)
        return output