import os
import Constants
import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile

class DataPreprocessor:
    def get_file_paths_and_labels(self, data_dir):
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
        mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=Constants.No_OF_MFCCs)
        return mfcc # nd array

    def extract_spectrogram(self, file_path):
        sampling_rate, samples = wavfile.read(file_path)
        _, _, spectrogram = signal.spectrogram(samples, sampling_rate) # The first 2 are arrays of frequencies and segment times respectively
        return spectrogram # nd array.  By default, the last axis of spectrogram corresponds to the segment times.




    


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    file_paths, labels = preprocessor.get_file_paths_and_labels()
    feature = preprocessor._mfcc_extractor(file_paths[-1])
