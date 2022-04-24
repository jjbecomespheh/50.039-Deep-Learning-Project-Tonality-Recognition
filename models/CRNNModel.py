import logging
import torch.nn as nn
import numpy as np


class CRNN(nn.Module):
    """
    CRNN class
    """
    def __init__(self, config=''):
        super(CRNN, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.classes = None

        cnn = nn.Sequential()

        
    def forward(self, *input):
        """
        Forward pass logic
        :return: Model output
        """
        



    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(CRNN, self).__str__() + '\nTrainable parameters: {}'.format(params)

    def nll_loss(output, target):
        # loss for log_softmax
        return F.nll_loss(output, target)

    def cross_entropy(output, target):
        return F.cross_entropy(output, target)


def RNN(X_shape, nb_classes):
    '''
    Implementing only the RNN
    '''
    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()

    model.add(Permute((time_axis, frequency_axis, channel_axis),
                      input_shape=input_shape))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))

    model.add(Dense(nb_classes))  # note sure about this
    model.add(Activation('softmax'))

    # Output layer
    return model