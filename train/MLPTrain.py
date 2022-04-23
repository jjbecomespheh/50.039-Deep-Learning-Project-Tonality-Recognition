
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import Constants
from TrainHelpers import mfcc_model_training_phase, mfcc_model_testing_phase, gen_confusion_matrix, data_prep
from models.MLPModel import MLP
from Dataset import TessDataset

def train_mlp(model_output_path, cf_path, learning_rate, early_stopping): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_val, x_test, y_train, y_val, y_test = data_prep("../data")
    train_loader = DataLoader(TessDataset(x_train, y_train), batch_size=Constants.MLP_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TessDataset(x_test, y_test), batch_size=Constants.MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TessDataset(x_val, y_val), batch_size=Constants.MLP_BATCH_SIZE, shuffle=True)
    mlp_model = MLP(Constants.MLP_INPUT_SIZE, Constants.MLP_HIDDEN_SIZE_1, Constants.MLP_HIDDEN_SIZE_2, Constants.MLP_OUTPUT_SIZE)
    mlp_model.to(device)
    tb = SummaryWriter()
    print('mlp_model: ', mlp_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), learning_rate)
    mfcc_model_training_phase(mlp_model, train_loader, val_loader, optimizer, criterion, tb, Constants.MLP_EPOCHS, Constants.MLP_ES_PATIENCE, Constants.MLP_BATCH_SIZE, early_stopping) 
    # python has pass by reference so the model gets updated
    mfcc_model_testing_phase(mlp_model, test_loader, model_output_path, Constants.MLP_BATCH_SIZE)
    gen_confusion_matrix(mlp_model, test_loader, cf_path)
    tb.close()

if __name__ == '__main__':
    train_mlp(f"./weights/MLP_NoES.pt", f"./cf/MLP_NoES.png", 0.001, False)
    train_mlp(f"./weights/MLP_ES.pt", f"./cf/MLP_ES.png", 0.001, True)
    for learning_rate in Constants.EXPT_LEARNING_RATES:
        train_mlp(f"./weights/MLP_ES_LR{learning_rate}.pt", f"./cf/MLP_ES_LR{learning_rate}.png", learning_rate, True)