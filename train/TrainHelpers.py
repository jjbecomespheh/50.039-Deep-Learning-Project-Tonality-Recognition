import sys
from os.path import dirname, abspath
import os
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import torch
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from Preprocessor import DataPreprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_accuracy(out, actual_labels, batch_size):
    predictions = out.max(dim=1)[1]
    correct = (predictions == actual_labels).sum().item()
    accuracy = correct/batch_size
    return accuracy

def mfcc_model_training_phase(model, train_loader, val_loader, optimizer, criterion, tb, epochs, patience, batch_size, early_stopping = True):
    print('Training started...')
    last_epoch_val_loss, trigger_count = math.inf, 0
    for epoch in range(epochs):
        train_loss, train_acc, batch_no = 0, 0, 0
        model.train()
        for batch in train_loader:
            batch_no += 1
            mfccs, labels = batch
            mfccs, labels = mfccs.to(device), labels.to(device)
            mfccs = torch.squeeze(mfccs)
            if epoch == 0 and batch_no == 0: tb.add_graph(model, mfccs)
            out = model(mfccs)
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_accuracy(out, labels, batch_size)

        # Start of Validation check. 
        current_epoch_train_loss, current_epoch_train_acc = train_loss/batch_no, train_acc/batch_no
        print('TRAIN | Epoch: {}/{} | Loss: {:.2f} | Accuracy: {:.2f}'.format(epoch + 1, epochs, current_epoch_train_loss, current_epoch_train_acc ))
        current_epoch_val_loss, current_epoch_val_acc = mfcc_model_validation_phase(model, val_loader, criterion, batch_size)
        print('VALIDATION | Epoch: {}/{} | Loss: {:.2f} | Accuracy: {:.2f}'.format(epoch + 1, epochs, current_epoch_val_loss, current_epoch_val_acc ))
        if early_stopping and current_epoch_val_loss > last_epoch_val_loss: 
            trigger_count += 1
            print(f"VALIDATION | Epoch {epoch + 1}/{epochs} | Current Loss: {np.round(current_epoch_val_loss, 2)} > Last Loss:  {np.round(last_epoch_val_loss, 2)} | Trigger Count: {trigger_count}")
            if trigger_count >= patience: 
                print(f"VALIDATION | Epoch {epoch + 1}/{epochs} | Early Stopping Here")
                return
        else: trigger_count = 0
        last_epoch_val_loss = current_epoch_val_loss

        lstm_tensorboard(tb, model, epoch, current_epoch_train_loss, current_epoch_val_loss, current_epoch_train_acc, current_epoch_val_acc)

def mfcc_model_validation_phase(model, val_loader, criterion, batch_size):
    val_loss, batch_no, val_acc = 0, 0, 0
    model.eval()

    for batch in val_loader:
        batch_no += 1
        mfccs, labels = batch
        mfccs, labels = mfccs.to(device), labels.to(device)
        mfccs = torch.squeeze(mfccs)
        out = model(mfccs)
        loss = criterion(out, labels)
        val_loss += loss.item()
        val_acc += get_accuracy(out, labels, batch_size)

    return val_loss / batch_no, val_acc / batch_no
        
def mfcc_model_testing_phase(model, test_loader, model_out_path, batch_size):
    print('Testing Started...')
    test_acc, batch_no = 0, 0
    model.eval()

    for batch in test_loader:
        batch_no += 1
        mfccs, labels = batch
        mfccs, labels = mfccs.to(device), labels.to(device)
        mfccs = torch.squeeze(mfccs)
        out = model(mfccs)
        test_acc += get_accuracy(out, labels, batch_size)

    print('TEST | Average Accuracy per {} Loaders: {:.5f}'.format(batch_no, test_acc/batch_no))
    torch.save(model.state_dict(), model_out_path)

def gen_confusion_matrix(model, test_loader, cf_path):
    print('Generating Confusion Matrix...')
    torch.no_grad()
    model.eval()
    preds, actuals = [], []

    for batch in test_loader:
        mfccs, labels = batch
        mfccs, labels = mfccs.to(device), labels.to(device)
        mfccs = torch.squeeze(mfccs)
        outputs = model(mfccs)
        outputs = outputs.max(dim=1)[1]
        for (output,label) in zip(outputs, labels):
            preds.append(output.cpu())
            actuals.append(label.cpu())
    cf = sklearn.metrics.confusion_matrix(preds, actuals)
    plt.figure(figsize=(16, 5))
    sns.heatmap(cf, cmap='icefire', annot=True, linewidths=0.1, fmt = ',')
    plt.title('Confusion Matrix: Model', fontsize=15)
    plt.savefig(cf_path)
    print('Confusion Matrix stored in ', cf_path)

def lstm_tensorboard(tb, model, epoch, current_epoch_train_loss, current_epoch_val_loss, current_epoch_train_acc, current_epoch_val_acc):
    tb.add_scalar("Training Loss", current_epoch_train_loss, epoch)
    tb.add_scalar("Training Accuracy", current_epoch_train_acc, epoch)
    tb.add_scalar("Validation Loss", current_epoch_val_loss, epoch)
    tb.add_scalar("Validation Accuracy", current_epoch_val_acc, epoch)
    for name, weight in model.named_parameters():
        tb.add_histogram(name, weight, epoch)
        tb.add_histogram(f"{name}.grad", weight.grad, epoch)

def data_prep(dir_path):
    files = os.listdir()
    if ('x_train.npy' in files) and ('x_test.npy' in files) and ('x_val.npy' in files)\
    and ('y_train.npy' in files) and ('y_test.npy' in files) and ('y_val.npy' in files):
        with open('x_train.npy', 'rb') as f:
            x_train = np.load(f)
        with open('x_val.npy', 'rb') as f:
            x_val = np.load(f)
        with open('x_test.npy', 'rb') as f:
            x_test = np.load(f)
        with open('y_train.npy', 'rb') as f:
            y_train = np.load(f)
        with open('y_val.npy', 'rb') as f:
            y_val = np.load(f)
        with open('y_test.npy', 'rb') as f:
            y_test = np.load(f)
    else:
        preprocessor = DataPreprocessor()
        x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.mfcc_data_prep(dir_path)
    return x_train, x_val, x_test, y_train, y_val, y_test