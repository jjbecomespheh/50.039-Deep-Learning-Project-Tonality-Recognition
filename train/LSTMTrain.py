from TrainHelpers import lstm_training_phase, lstm_testing_phase, gen_confusion_matrix
from models.LSTMModel import LSTM
from Preprocessor import DataPreprocessor
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import Constants
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


def train_lstm(model_output_path, cf_path):
    preprocessor = DataPreprocessor()
    x_train, _, x_test, y_train, _, y_test = preprocessor.mfcc_data_prep("../data")
    train_loader = DataLoader(TessDataset(x_train, y_train), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TessDataset(x_test, y_test), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
    lstm_model = LSTM(Constants.LSTM_INPUT_SIZE, Constants.LSTM_HIDDEN_SIZE, Constants.LSTM_LAYER_SIZE, Constants.LSTM_OUTPUT_SIZE)
    print('lstm_model: ', lstm_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=Constants.LSTM_LEARNING_RATE)
    lstm_training_phase(lstm_model, train_loader, optimizer, criterion)
    lstm_testing_phase(lstm_model, test_loader, model_output_path)
    gen_confusion_matrix(lstm_model, test_loader, cf_path)
	

if __name__ == '__main__':
    train_lstm("./weights/LSTMModel.pt", "./cf/LSTMModel.png")
