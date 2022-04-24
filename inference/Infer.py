import pandas as pd
import numpy as np

import os
import sys
from tqdm import tqdm

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
from torchinfo import summary
from train.MelTrainHelper import TrainHelper

from Preprocessor import *
import Constants

def infer(model, path, all_mel_specs):
    EMOTIONS= {0:'neutral',1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'ps'}
    n=len(path)
    train_helper = TrainHelper()
    preprocessor = DataPreprocessor()
    if len(path) == 1:
        path = path[0]
        audio, sample_rate = librosa.load(path, duration=3, offset=0.5, sr=48000)
        audio_signal = np.zeros((int(sample_rate*3,)))
        audio_signal[:len(audio)] = audio
        mel_specs = preprocessor.extract_mel_spectogram(audio_signal)
        X_test = mel_specs
        X_test = np.expand_dims(X_test, axis=0)
        X_test = np.concatenate((X_test, all_mel_specs))
        X_test = preprocessor.reshape_scale_data(X_test)
        X_test = X_test[:1]
    else:
        audio_signals = preprocessor.extract_audio_signals(path)
        mel_specs = preprocessor.extract_mel_spectograms(audio_signals)
        mel_specs = np.concatenate((mel_specs, all_mel_specs))
        X_test = preprocessor.reshape_scale_data(mel_specs)
        X_test = X_test[:len(path)]
    Y_test = [1]*n
    X_test_tensor = torch.tensor(X_test,device='cpu').float()
    Y_test_tensor = torch.tensor(Y_test,dtype=torch.long,device='cpu')
    test_loss, test_acc, predictions, output_softmax = train_helper.validate_top3(X_test_tensor,Y_test_tensor,model)
    
    if len(predictions.tolist()) == 1:
        ground_truth = path.split("_")[-1].split(".")[-2]
        top3_prob, top3 = torch.topk(output_softmax, 3)
        top3 = top3.detach().numpy()
        top3_prob = top3_prob.detach().numpy()
        print("Audio File: ", path)
        print("Predicted Emotions: ",EMOTIONS[top3[0,0]], "\t| Ground Truth: ", ground_truth)
        print("Top 1: ", EMOTIONS[top3[0,0]], "Prob: ", round(top3_prob[0,0]*100, 3),"%", "\t| Top 2: ", EMOTIONS[top3[0,1]], "Prob: ", round(top3_prob[0,1]*100, 3),"%", "\t| Top 3: ", EMOTIONS[top3[0,2]], "Prob: ", round(top3_prob[0,2]*100, 3),"%")
    else:
        print("\n")
        i = 0 
        for pred, file_name in zip(predictions.tolist(), path):
            ground_truth = file_name.split("_")[-1].split(".")[-2]
            
            top3_prob, top3 = torch.topk(output_softmax[i], 3)
            top3 = top3.detach().numpy()
            top3_prob = top3_prob.detach().numpy()
            i += 1
            print("Audio File: ", file_name)
            print("Predicted Emotions: ",EMOTIONS[top3[0]], "\t| Ground Truth: ", ground_truth)
            print("Top 1:", EMOTIONS[top3[0]], "(Prob:", round(top3_prob[0]*100, 2),"%)", "\t| Top 2:", EMOTIONS[top3[1]], "(Prob:", round(top3_prob[1]*100, 2),"%)", "\t| Top 3:", EMOTIONS[top3[2]], "(Prob:", round(top3_prob[2]*100, 2),"%)\n")
    return