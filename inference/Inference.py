import torch
import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import Constants 
from models.LSTMModel import LSTM
from models.MLPModel import MLP
from Preprocessor import DataPreprocessor

def mfccs_infer(model_name, model_weights_path, wav_file_paths):
    if model_name == 'lstm':
        model = LSTM(Constants.LSTM_INPUT_SIZE, Constants.LSTM_HIDDEN_SIZE, Constants.LSTM_LAYER_SIZE, Constants.LSTM_OUTPUT_SIZE, Constants.LSTM_DROPOUT)
    if model_name == 'mlp':
        model = MLP(Constants.MLP_INPUT_SIZE, Constants.MLP_HIDDEN_SIZE_1, Constants.MLP_HIDDEN_SIZE_2, Constants.MLP_OUTPUT_SIZE)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    mfccs = DataPreprocessor().extract_mfccs(wav_file_paths)
    mfccs = torch.from_numpy(mfccs)
    mfccs = torch.squeeze(mfccs)
    out = model(mfccs)
    prediction = out.max(dim=1)[1]
    return prediction

if __name__ == "__main__":
    test_files = ['./TestData/Test1.wav', './TestData/Test2.wav', './TestData/Test3.wav']
    # predictions = mfccs_infer("lstm", "../train/weights/LSTM_ES4_D05_Bi.pt", test_files)
    predictions = mfccs_infer("mlp", "../train/weights/MLP_ES4.pt", test_files)
    print(predictions)