from dataclasses import dataclass
import sys
from os.path import dirname, abspath

from train.TrainHelpers import lstm_testing_phase
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import Constants
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Preprocessor import DataPreprocessor
from models.LSTMModel import LSTM
from TrainHelpers import lstm_training_phase, lstm_testing_phase

class TessDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def get_accuracy(out, actual_labels):
    predictions = out.max(dim=1)[1]
    correct = (predictions == actual_labels).sum().item()
    accuracy = correct/Constants.LSTM_BATCH_SIZE
    return accuracy

def train_lstm():
  preprocessor = DataPreprocessor()
  x_train, _, x_test, y_train, _, y_test = preprocessor.mfcc_data_prep("../data")
  train_loader = DataLoader(TessDataset(x_train, y_train), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(TessDataset(x_test, y_test), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
  lstm_model = LSTM(Constants.LSTM_INPUT_SIZE, Constants.LSTM_HIDDEN_SIZE, Constants.LSTM_LAYER_SIZE, Constants.LSTM_OUTPUT_SIZE)
  print('lstm_model: ', lstm_model)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(lstm_model.parameters(), lr=Constants.LSTM_LEARNING_RATE)
  lstm_training_phase(lstm_model, train_loader, optimizer, criterion)
  lstm_testing_phase(lstm_model, test_loader)

if __name__ == '__main__':
    train_lstm()