import matplotlib.pyplot as plt
import numpy as np
import time

def generate_plot(train, val, filename, typ='loss', model='MLP', savepath='figs/'):
    if len(train) != len(val):
        raise Exception("Input data arrays of different length.")
    
    # params
    nepochs = len(train)
    title_fontsize = '18'
    label_fontsize = '15'
    xtick_fontsize = '14'
    ytick_fontsize = '14'
    
    fig = plt.figure(figsize=(8,6))
    x = np.arange(nepochs)
    
    plt.plot(x, train, label='train')
    plt.plot(x, val, label='val')
    
    if typ == 'loss':
        title = 'Training and Validation Loss'
        ylabel = 'Loss Value'
    elif typ == 'acc':
        title = 'Training and Validation Accuracy'
        ylabel = 'Accuracy'
        
    title = title + ' ' + '(' + model + ')'
        
    xlabel = 'Epochs'
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.legend(fontsize=ytick_fontsize, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(savepath + filename + '.png', dpi=250)
    # plt.show()

if __name__ == "__main__":
    train = np.genfromtxt('csv/dummy01.csv', delimiter=',')
    val = np.genfromtxt('csv/dummy02.csv', delimiter=',')

    filename = 'LSTM_01'
    generate_plot(train, val, filename, typ='acc', model='LSTM')