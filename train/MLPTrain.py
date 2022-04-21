
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import Constants
from Preprocessor import DataPreprocessor
from LSTMTrainHelpers import mfcc_model_training_phase, mfcc_model_testing_phase, gen_confusion_matrix
from models.MLPModel import MLP
from Dataset import TessDataset

def train_lstm(model_output_path, cf_path): 
    preprocessor = DataPreprocessor()
    x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.mfcc_data_prep("../data")
    train_loader = DataLoader(TessDataset(x_train, y_train), batch_size=Constants.MLP_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TessDataset(x_test, y_test), batch_size=Constants.MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TessDataset(x_val, y_val), batch_size=Constants.MLP_BATCH_SIZE, shuffle=True)
    mlp_model = MLP(Constants.MLP_INPUT_SIZE, Constants.MLP_HIDDEN_SIZE_1, Constants.MLP_HIDDEN_SIZE_2, Constants.MLP_OUTPUT_SIZE)
    tb = SummaryWriter()
    print('mlp_model: ', mlp_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=Constants.MLP_LEARNING_RATE)
    mfcc_model_training_phase(mlp_model, train_loader, val_loader, optimizer, criterion, tb) # python has pass by reference so the model gets updated
    mfcc_model_testing_phase(mlp_model, test_loader, model_output_path)
    gen_confusion_matrix(mlp_model, test_loader, cf_path)
    tb.close()

if __name__ == '__main__':
    train_lstm("./weights/MLPModel_ESDropout_BiDir.pt", "./cf/MLPModel_ESDropout_BiDir.png")
