from dataclasses import dataclass
import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import Constants
import torch
import numpy as np
from Preprocessor import DataPreprocessor

class TessDataset(torch.utils.data.Dataset):
    def __init__(self):
        preprocessor = DataPreprocessor()
        x_train, _, _, y_train, _, _ = preprocessor.mfcc_data_prep("./data")
        self.x = x_train
        self.y = y_train
        self.n_samples = x_train.shape[0]

    def __len__(self):
        self.n_samples = self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]



if __name__ == '__main__':
    print(Constants.LSTM_BATCH_SIZE)