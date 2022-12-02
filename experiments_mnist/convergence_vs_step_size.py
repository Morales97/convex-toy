import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from mnist_log_regression import train_mnist

config = {
    'loss_th': 0.33,
    # 'train_loss_th': None, #0.25,
    # 'acc_th': 0.915,
    'data_distr': 'iid',     # 'iid' or 'non-iid'
    'epochs': 30,
    'steps': 30*58,
    # 'batch_size': 13,
    # 'batch_size': 128,
    # 'n_nodes': 100,
    'batch_size': 32,
    'n_nodes': 32,
    'eval': True,
    'path': 'data/mnist',
    'log': True,
    # 'lr_factors': np.logspace(np.log10(0.5),np.log10(6), 12)
    'lr_factors': np.linspace(0.5, 6, 12),
    'net': 'log_reg',
}
expts = [
    # {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 3 local steps', 'local_steps': 3},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 29 local steps', 'local_steps': 29},
    {'topology': 'fully_connected', 'label': 'Fully connected, 58 local steps', 'local_steps': 58},
    # {'topology': 'solo', 'label': 'Solo', 'local_steps': 0},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 4, 'label': 'random (degree: 4)', 'local_steps': 0},
    # {'topology': 'ring',  'label': 'Ring', 'local_steps': 0},
]


def sweep_step_sizes(config, expt):
    arr_epochs = []
    arr_ECE = []
    arr_test_loss = []
    for lr_factor in config['lr_factors']:
        config['lr_factor'] = lr_factor
        accuracies, test_loss, train_loss, ECE = train_mnist(config, expt)
        epochs = len(accuracies)
        # steps = len(train_loss)
        if epochs == config['epochs']:
            break 
        arr_epochs.append(epochs)
        arr_ECE.append(ECE)
        arr_test_loss.append(test_loss)

    return arr_epochs, arr_ECE, arr_test_loss

def plot_convergence_vs_lr(ax, losses, lr_factors, label=None):
    x = lr_factors[:len(losses)]
    ax.plot(x, losses, label=label)
    ax.set_xlabel('LR factor (base LR * factor)')
    ax.set_ylabel('Epochs until conv')


def plot_ECE_vs_lr(ax, arr_ECE, lr_factors, label=None):
    x = lr_factors[:len(arr_ECE)]
    ax.plot(x, arr_ECE, label=label)
    ax.set_xlabel('LR factor (base LR * factor)')
    ax.set_ylabel('ECE')


def plot_test_loss_vs_lr(ax, arr_test_loss, lr_factors, label=None):
    x = lr_factors[:len(arr_test_loss)]
    ax.plot(x, arr_test_loss, label=label)
    ax.set_xlabel('LR factor (base LR * factor)')
    ax.set_ylabel('Test loss')


def plot_all():
    ''' plot epochs till convergence, final test loss and ECE'''

    fig, ax = plt.subplots(1,3, figsize=(16,5))
    for i in range(len(expts)):
        arr_epochs, arr_ECE, arr_test_loss = sweep_step_sizes(config, expts[i])
        plot_convergence_vs_lr(ax[0], arr_epochs, config['lr_factors'], expts[i]['label'])
        plot_ECE_vs_lr(ax[1], arr_ECE, config['lr_factors'], expts[i]['label'])
        plot_test_loss_vs_lr(ax[2], arr_test_loss, config['lr_factors'], expts[i]['label'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_all()
    ax = plt.subplot(111)
    for i in range(len(expts)):
        arr_epochs, arr_ECE, arr_test_loss = sweep_step_sizes(config, expts[i])
        plot_convergence_vs_lr(ax, arr_epochs, config['lr_factors'], expts[i]['label'])
        # plot_ECE_vs_lr(ax, arr_ECE, config['lr_factors'], expts[i]['label'])
    plt.legend()
    plt.show()