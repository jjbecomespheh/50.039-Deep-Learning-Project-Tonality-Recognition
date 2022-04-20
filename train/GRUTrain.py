from models.GRUModel import GRU
from Preprocessor import DataPreprocessor
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import Constants
from dataclasses import dataclass
import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)


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

def train():
    preprocessor = DataPreprocessor()
    x_train, _, x_test, y_train, _, y_test = preprocessor.mfcc_data_prep("../data")
    train_loader = DataLoader(TessDataset(x_train, y_train), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TessDataset(x_test, y_test), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
    model = GRU(Constants.LSTM_INPUT_SIZE, Constants.LSTM_HIDDEN_SIZE, Constants.LSTM_LAYER_SIZE, Constants.LSTM_OUTPUT_SIZE)
    print('gru_model: ', model)
    train_network(model, train_loader, test_loader)

def train_network(model, train_loader, test_loader, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('Training started...')
    # Train the data multiple times
    for epoch in range(Constants.LSTM_EPOCHS):
        train_loss = 0
        train_acc = 0
        model.train()
        batch_no = 0
        for batch in train_loader:
            batch_no += 1
            mfccs, labels = batch
            mfccs = torch.squeeze(mfccs)
            out = model(mfccs)
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_accuracy(out, labels)
        print('TRAIN | Epoch: {}/{} | Loss: {:.2f} | Accuracy: {:.2f}'.format(epoch + 1, Constants.LSTM_EPOCHS, train_loss/batch_no, train_acc/batch_no))
    print('Testing Started...')
    test_acc = 0
    model.eval()
    batch_no = 0

    for batch in test_loader:
        batch_no += 1
        mfccs, labels = batch
        mfccs = torch.squeeze(mfccs)
        out = model(mfccs)
        test_acc += get_accuracy(out, labels)

    # Print Final Test Accuracy
    print('TEST | Average Accuracy per {} Loaders: {:.5f}'.format(
        batch_no, test_acc/batch_no))
    torch.save(model.state_dict(), "GRUModel.pt")


if __name__ == '__main__':
    train()