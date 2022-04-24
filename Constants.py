# For Generating MFCC and spectograms
SAMPLING_RATE = 22050  # Librosa
NO_OF_MFCCs = 40  # No of MFCCs returned by librosa

# For Generating Train-val-test split
TRAIN_SET_SIZE = 0.8 # 80%
TEST_SET_SIZE = 0.5 # 50% of the remaining data after extracting training set --> 10%
RANDOM_STATE = 20 # For shuffle in train_test_split. Need this to stratify the split.

# For LSTM Model
LSTM_BATCH_SIZE = 10
LSTM_INPUT_SIZE = 40
LSTM_HIDDEN_SIZE = 100
LSTM_LAYER_SIZE = 2
LSTM_OUTPUT_SIZE = 7
LSTM_DROPOUT = 0.5
LSTM_EPOCHS = 250
LSTM_LEARNING_RATE = 0.001
LSTM_ES_PATIENCE = 4

# For MLP Model
MLP_BATCH_SIZE = 10
MLP_INPUT_SIZE = 40
MLP_HIDDEN_SIZE_1 = 40
MLP_HIDDEN_SIZE_2 = 40
MLP_OUTPUT_SIZE = 7
MLP_DROPOUT = 0.5
MLP_EPOCHS = 250
MLP_LEARNING_RATE = 0.001
MLP_ES_PATIENCE = 4

# For CNN and CNN-LSTM
CNN_SAMPLING_RATE = 48000
EXPT_LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
EXPT_DROPOUTS = [0.1, 0.2, 0.3, 0.4, 0.5]