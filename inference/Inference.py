import torch
import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

from models.LSTMModel import LSTM
from models.MLPModel import MLP
from Preprocessor import DataPreprocessor

def mfccs_infer(model_name, model_weights_path, wav_file_paths):
    if model_name == 'lstm':
        model = LSTM()
    if model_name == 'mlp':
        model = MLP()
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    mfccs = DataPreprocessor().extract_mfccs(wav_file_paths)
    mfccs = torch.squeeze(mfccs)
    out = model(mfccs)
    prediction = out.max(dim=1)[1]
    return prediction
    

