import torch
import sys
import os
from os.path import dirname, abspath
import argparse
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import Constants 
from models.LSTMModel import LSTM
from models.MLPModel import MLP
from Preprocessor import DataPreprocessor


def mfccs_infer(model_name, model_weights_path, wav_file_paths, dropout =0.4, bidirectional = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'lstm':
        model = LSTM(Constants.LSTM_INPUT_SIZE, Constants.LSTM_HIDDEN_SIZE, Constants.LSTM_LAYER_SIZE, Constants.LSTM_OUTPUT_SIZE, dropout, bidirectional)
    if model_name == 'mlp':
        model = MLP(Constants.MLP_INPUT_SIZE, Constants.MLP_HIDDEN_SIZE_1, Constants.MLP_HIDDEN_SIZE_2, Constants.MLP_OUTPUT_SIZE)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    mfccs = DataPreprocessor().extract_mfccs(wav_file_paths)
    mfccs = torch.from_numpy(mfccs)
    mfccs = torch.squeeze(mfccs, 3)
    out = model(mfccs)
    prediction = out.max(dim=1)[1]
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--Model", help="Model type used for inference --> either lstm or mlp", type=str, required=True)
    parser.add_argument("-w", "--Weights", help="Weights used for inference --> enter a valid path", type=str, required=True)
    parser.add_argument("-d", "--Dropout", help="Dropout of the model used for inference", type=float, required=False)
    parser.add_argument("-b", "--Bidirectional", help="Bidircetional attribute of the model used for inference", type=bool, required=False)
    args = parser.parse_args()

    if not (args.Model == 'lstm' or args.Model == "mlp"):
        print("Please enter a valid model type --> Options are either 'lstm' or 'mlp'")
    elif not os.path.exists(args.Weights):
        print("Please enter a valid path for the weights of the model")
    else:
        test_data_dir = "./TestData"
        test_data_dir_files = os.listdir(test_data_dir)
        test_files = [test_data_dir + "/" + file for file in test_data_dir_files if file.endswith(".wav")]
        predictions = mfccs_infer(args.Model, args.Weights, test_files)
        mffc_label_emotion_mapping = { 0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "ps", 6: "sad" }
        mapped_predictions = [mffc_label_emotion_mapping[int(prediction)] for prediction in predictions.tolist()]
        for test_file, mapped_prediction in zip(test_files, mapped_predictions):
            print(f"File: {test_file}, Prediction: {mapped_prediction}")