import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from mnist_log_regression import train_mnist

config = {
    'data_distr': 'iid',     # 'iid' or 'non-iid'
    # 'steps': 200,
    # 'steps_log': 50,
    'epochs': 200,
    # 'batch_size': 13,
    'batch_size': 128,
    'n_nodes': 100,
    'eval': True,
    'path': 'data/mnist',
    'log': True,
    'lr_factors': np.linspace(0.1, 5, 14),
}
expts = [
    {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 4, 'label': 'random (degree: 4)', 'local_steps': 0},
    {'topology': 'ring',  'label': 'Ring', 'local_steps': 0},
]


def sweep_step_sizes(config, expt):
    losses = []
    for lr_factor in config['lr_factors']:
        config['lr_factor'] = lr_factor
        accuracies, loss = train_mnist(config, expt)
        if loss > 0.5:
            break 
        losses.append(loss)
    return losses


def plot_loss_vs_lr(losses, lr_factors, label=None):
    x = lr_factors[:len(losses)]
    plt.plot(x, losses, label=label)
    plt.xlabel('LR factor (base LR * factor)')
    plt.ylabel('Test Loss')

if __name__ == '__main__':
    for i in range(len(expts)):
        losses = sweep_step_sizes(config, expts[i])
        plot_loss_vs_lr(losses, config['lr_factors'], expts[i]['label'])
    plt.legend()
    plt.show()