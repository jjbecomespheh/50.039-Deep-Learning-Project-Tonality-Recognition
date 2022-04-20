import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import torch
import Constants

def get_accuracy(out, actual_labels):
    predictions = out.max(dim=1)[1]
    correct = (predictions == actual_labels).sum().item()
    accuracy = correct/Constants.LSTM_BATCH_SIZE
    return accuracy

def lstm_training_phase(model, train_loader, optimizer, criterion):
    print('Training started...')
    for epoch in range(Constants.LSTM_EPOCHS):
        train_loss, train_acc, batch_no = 0, 0, 0
        model.train()
        for batch in train_loader:
            batch_no += 1
            mfccs, labels = batch
            mfccs = torch.squeeze(mfccs)
            out = model(mfccs)
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_accuracy(out, labels)
        print('TRAIN | Epoch: {}/{} | Loss: {:.2f} | Accuracy: {:.2f}'.format(epoch + 1, Constants.LSTM_EPOCHS, train_loss/batch_no, train_acc/batch_no))

def lstm_testing_phase(model, test_loader, model_out_path):
    print('Testing Started...')
    test_acc, batch_no = 0, 0
    model.eval()

    for batch in test_loader:
        batch_no += 1
        mfccs, labels = batch
        mfccs = torch.squeeze(mfccs)
        out = model(mfccs)
        test_acc += get_accuracy(out, labels)

    print('TEST | Average Accuracy per {} Loaders: {:.5f}'.format(batch_no, test_acc/batch_no))
    torch.save(model.state_dict(), model_out_path)
