import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from mnist_log_regression import train_mnist

config = {
    # 'loss_th': 0.33,
    # 'train_loss_th': None, #0.25,
    'acc_th': 0.915,
    'data_distr': 'iid',     # 'iid' or 'non-iid'
    # 'steps': 200,
    # 'steps_log': 50,
    'epochs': 50,
    # 'batch_size': 13,
    # 'batch_size': 128,
    # 'n_nodes': 100,
    'batch_size': 32,
    'n_nodes': 32,
    'eval': True,
    'path': 'data/mnist',
    'log': True,
    # 'lr_factors': np.logspace(np.log10(0.5),np.log10(6), 12)
    'lr_factors': np.linspace(0.5, 6, 20),
}
expts = [
    # {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0},
    {'topology': 'fully_connected', 'label': 'Fully connected, 3 local steps', 'local_steps': 3},
    {'topology': 'fully_connected', 'label': 'Fully connected, 29 local steps', 'local_steps': 29},
    {'topology': 'fully_connected', 'label': 'Fully connected, 58 local steps', 'local_steps': 58},
    # {'topology': 'solo', 'label': 'Solo', 'local_steps': 0},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 4, 'label': 'random (degree: 4)', 'local_steps': 0},
    # {'topology': 'ring',  'label': 'Ring', 'local_steps': 0},
]


def sweep_step_sizes(config, expt):
    arr = []
    for lr_factor in config['lr_factors']:
        config['lr_factor'] = lr_factor
        accuracies, _, train_loss = train_mnist(config, expt)
        epochs = len(accuracies)
        # steps = len(train_loss)
        if epochs == config['epochs']:
        # if steps == config['epochs'] * int(60000 // (config['n_nodes']*config['batch_size'])):
            break 
        arr.append(epochs)
        # arr.append(steps)
    return arr


def plot_convergence_vs_lr(losses, lr_factors, label=None):
    x = lr_factors[:len(losses)]
    plt.plot(x, losses, label=label)
    plt.xlabel('LR factor (base LR * factor)')
    plt.ylabel('Epochs until conv')

if __name__ == '__main__':
    for i in range(len(expts)):
        arr = sweep_step_sizes(config, expts[i])
        plot_convergence_vs_lr(arr, config['lr_factors'], expts[i]['label'])
    plt.legend()
    plt.show()