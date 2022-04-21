
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import Constants
from Preprocessor import DataPreprocessor
from TrainHelpers import mfcc_model_training_phase, mfcc_model_testing_phase, gen_confusion_matrix
from models.LSTMModel import LSTM
from Dataset import TessDataset

def train_lstm(model_output_path, cf_path): 
    preprocessor = DataPreprocessor()
    x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.mfcc_data_prep("../data")
    train_loader = DataLoader(TessDataset(x_train, y_train), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TessDataset(x_test, y_test), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TessDataset(x_val, y_val), batch_size=Constants.LSTM_BATCH_SIZE, shuffle=True)
    lstm_model = LSTM(Constants.LSTM_INPUT_SIZE, Constants.LSTM_HIDDEN_SIZE, Constants.LSTM_LAYER_SIZE, Constants.LSTM_OUTPUT_SIZE, Constants.LSTM_DROPOUT)
    tb = SummaryWriter()
    print('lstm_model: ', lstm_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=Constants.LSTM_LEARNING_RATE)
    mfcc_model_training_phase(lstm_model, train_loader, val_loader, optimizer, criterion, tb, Constants.LSTM_EPOCHS, Constants.LSTM_ES_PATIENCE, Constants.LSTM_BATCH_SIZE) 
	# python has pass by reference so the model gets updated
    mfcc_model_testing_phase(lstm_model, test_loader, model_output_path, Constants.LSTM_BATCH_SIZE)
    gen_confusion_matrix(lstm_model, test_loader, cf_path)
    tb.close()

if __name__ == '__main__':
    train_lstm(f"./weights/LSTM_ES{Constants.LSTM_EPOCHS}_D05_Bi.pt", f"./cf/LSTM_ES{Constants.LSTM_EPOCHS}_D05_Bi.png")
