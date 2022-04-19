SAMPLING_RATE = 22050  # Librosa
NO_OF_MFCCs = 40  # No of MFCCs returned by librosa
TRAIN_SET_SIZE = 0.8 # 80%
TEST_SET_SIZE = 0.5 # 50% of the remaining data after extracting training set --> 10%
RANDOM_STATE = 20 # For shuffle in train_test_split. Need this to stratify the split.