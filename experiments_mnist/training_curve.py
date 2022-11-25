import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from mnist_log_regression import train_mnist



def plot_test_accuracy_vs_epoch(config, expt, accuracies):
    epochs = np.arange(config['epochs'])
    assert len(epochs) == len(accuracies)
    
    plt.plot(epochs, accuracies, label=expt['label'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')

def plot_train_loss_vs_epoch(config, expt, train_loss):
    epochs = np.arange(config['epochs'])
    # steps = np.arange(config['epochs'] * int(60000 // (config['n_nodes']*config['batch_size'])))
    # assert len(steps) == len(train_loss)
    train_loss = np.array(train_loss)
    train_loss = train_loss.reshape(-1, int(60000 // (config['n_nodes']*config['batch_size'])))
    train_loss_avg = np.mean(train_loss, axis=1)
    assert len(epochs) == len(train_loss_avg)
    # plt.plot(steps, train_loss, label=expt['label'])
    plt.plot(epochs, train_loss_avg, label=expt['label'])
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')

config = {
    'data_distr': 'iid',     # 'iid' or 'non-iid'
    'epochs': 40,
    'batch_size': 32,
    'n_nodes': 32,
    'eval': True,
    'path': 'data/mnist',
    'log': True,
    # 'lr_factor': 1,
    'lr_factor': 4,
}
expts = [
    # {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'centralized', 'label': 'Fully connected, lmbda 1e-3', 'local_steps': 0, 'lambda': 1e-3},
    # {'topology': 'centralized', 'label': 'Fully connected, lmbda 1e-4', 'local_steps': 0, 'lambda': 1e-4},
    # {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 3 local steps', 'local_steps': 3},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 10 local steps', 'local_steps': 10},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 18 local steps', 'local_steps': 18},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 1160 local steps', 'local_steps': 1160},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 2320 local steps', 'local_steps': 2320},
    {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    {'topology': 'exponential_graph', 'label': 'Exponential graph, 58 local steps', 'local_steps': 58},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 4, 'label': 'random (degree: 4)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 9, 'label': 'random (degree: 9)', 'local_steps': 0},
]



if __name__ == '__main__':
    for i in range(len(expts)):
        accuracies, loss_test, loss_train = train_mnist(config, expts[i])
        plot_test_accuracy_vs_epoch(config, expts[i], accuracies)
        # plot_train_loss_vs_epoch(config, expts[i], loss_train)
    plt.legend()
    plt.show()