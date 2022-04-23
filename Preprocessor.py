import os
import Constants
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from scipy.io import wavfile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataPreprocessor:
    def get_file_paths_and_labels(self, data_dir):
        # The dir structure is 
        # ./data  -> "TESS Toronto emotional speech set data" -> [OAF_angry, OAF_sad ..]
        file_paths, labels = list(), list()
        for dir_path, _, file_names in os.walk(data_dir):
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                file_paths.append(file_path)
                parts = file_name.split('_')
                label = parts[-1].split('.')[0].lower()
                labels.append(label)
        return file_paths, labels 

    def extract_mfcc(self, file_path): # MFCC = Mel-frequency Cepstral Coefficients 
        audio, sampling_rate = librosa.load(path = file_path, sr = Constants.SAMPLING_RATE, mono = True)
        mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=Constants.NO_OF_MFCCs)
        mean = np.mean(mfcc, axis=1)
        out_shape = (Constants.NO_OF_MFCCs, Constants.NO_OF_MFCCs)
        out = np.zeros(out_shape, dtype=np.float32)
        out[:, 0] = mean
        return out # array of shape (no of MFCCs, no of MFCCs)
        # return np.mean(mfcc, axis=1) #x array of shape (no of MFCCs, 1)

    def extract_mfccs(self, file_paths):
        mfccs = list()
        for file_path in tqdm(file_paths):
            mfccs.append(self.extract_mfcc(file_path))
        out = np.array(mfccs)
        out = np.expand_dims(out, axis = -1)
        return out # array of shape (no of audio files, no of MFCCs, 1)

    def convert_labels_to_OHE(self, labels): # labels --> an array containing labels 
        data = np.array(labels).reshape(-1, 1)
        one_hot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = one_hot_encoder.fit_transform(data)
        classes = one_hot_encoder.categories_ [0]
        return onehot_encoded, classes # onehot_encoded --> array of shape (no of audio files, no of classes)

    def convert_labels_to_LE(self, labels):
        data = np.array(labels).reshape(-1, 1)
        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(data)
        classes = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print("Classes are ", classes)
        return label_encoded, classes

    def train_val_test_split(self, x, y): # x, y --> arrays of same shape
        x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size = Constants.TRAIN_SET_SIZE, random_state= Constants.RANDOM_STATE, stratify=y)
        x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size = Constants.TEST_SET_SIZE, random_state= Constants.RANDOM_STATE, stratify=y_rem)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def mfcc_data_prep(self, data_dir): # combines the above functions for
        file_paths, labels = self.get_file_paths_and_labels(data_dir)
        mfccs = self.extract_mfccs(file_paths)
        le_classes, _ = self.convert_labels_to_LE(labels)
        x_train, x_val, x_test, y_train, y_val, y_test = self.train_val_test_split(mfccs, le_classes)
        return x_train, x_val, x_test, y_train, y_val, y_test