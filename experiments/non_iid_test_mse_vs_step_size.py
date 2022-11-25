import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt

sys.path.insert(0,  os.path.join(sys.path[0], '..'))
from linear_regression_DSGD import train

config = {
    'mse_th': None,
    'steps': 200,
    'num_clients': 16,
    'hidden_dim': 100,
    'eval': True,
    'path': 'data/linear_regression_distr_var0.001',
    'log': False,
    'lr_factors': np.linspace(0.1, 4, 30),
}

expts_base = {'local_steps': 0}
expts = [
    {'topology': 'fully_connected', 'label': 'Fully connected', **expts_base},
    {'topology': 'exponential_graph', 'label': 'Exponential Graph', **expts_base},
    #{'topology': 'EG_time_varying', 'label': 'time varying EG', **expts_base},
    #{'topology': 'EG_multi_step', 'label': 'multi step EG', **expts_base},
    {'topology': 'ring', 'label': 'Ring', **expts_base},
]

def sweep_step_sizes(config, expt):
    mse = []
    for lr_factor in config['lr_factors']:
        config['lr_factor'] = lr_factor
        mse_test = train(config, expt)
        if mse_test > 0.1:
            break 
        mse.append(mse_test)
    return mse


def plot_mse_vs_lr(mse, lr_factors, label=None):
    x = lr_factors[:len(mse)]
    plt.plot(x, mse, label=label)
    plt.xlabel('LR factor (1/L * factor)')
    plt.ylabel('Test MSE')

if __name__ == '__main__':
    for i in range(len(expts)):
        mse = sweep_step_sizes(config, expts[i])
        plot_mse_vs_lr(mse, config['lr_factors'], expts[i]['label'])
    plt.legend()
    plt.show()