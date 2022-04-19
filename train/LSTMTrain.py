from dataclasses import dataclass
import sys
from os.path import dirname, abspath
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

class TessDataset(Dataset):
    def __init__(self):
        preprocessor = DataPreprocessor()
        x_train, _, _, y_train, _, _ = preprocessor.mfcc_data_prep("../data")
        self.x = x_train
        self.y = y_train
        self.n_samples = x_train.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def train_lstm():
  lstm_model = LSTM(Constants.LSTM_INPUT_SIZE, Constants.LSTM_HIDDEN_SIZE, Constants.LSTM_LAYER_SIZE, Constants.LSTM_OUTPUT_SIZE)
  print('lstm_model: ', lstm_model)
  train_loader = DataLoader(TessDataset(), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
  train_network(lstm_model, train_loader)
  

def train_network(model, train_loader, learning_rate=0.01):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  print('Training started...')
  # Train the data multiple times
  for epoch in range(Constants.LSTM_EPOCHS):
    print("Epoch: {}".format(epoch))
    train_loss = 0
    train_acc = 0
    model.train()
    batch_no = 0
    for batch in train_loader:
      batch_no+=1
      mfccs, labels = batch
      mfccs = torch.squeeze(mfccs)
      out = model(mfccs)
      print(out)
      optimizer.zero_grad()
      loss = criterion(out, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    print('TRAIN | Epoch: {}/{} | Loss: {:.2f}'.format(epoch+1, Constants.LSTM_EPOCHS, train_loss/batch_no))
    


if __name__ == '__main__':
    train_lstm()