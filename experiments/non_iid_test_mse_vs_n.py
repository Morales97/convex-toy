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
    'lr_factor': 1,
}

expts_base = {'local_steps': 0}
expts = [
    {'topology': 'fully_connected', 'label': 'Fully connected', **expts_base},
    {'topology': 'exponential_graph', 'label': 'Exponential Graph', **expts_base},
    #{'topology': 'EG_time_varying', 'label': 'time varying EG', **expts_base},
    #{'topology': 'EG_multi_step', 'label': 'multi step EG', **expts_base},
    {'topology': 'ring', 'label': 'Ring', **expts_base},
]

def increase_n_keep_lr_test_mse_old(config, expt):
    mse = []
    for i in range(config['num_clients']):
        mse_test = train(config, expt, n_clients=i+1)
        mse += [mse_test]
        print('Workers: %d\tTest MSE: %.5f'%(i+1, mse_test))
    return mse

def increase_n_keep_lr_test_mse(config, expt):
    mse = []
    max_n_clients = config['num_clients']
    for i in range(max_n_clients):
        expt['num_clients'] = i+1
        mse_test = train(config, expt)
        mse += [mse_test]
        print('Workers: %d\tTest MSE: %.5f'%(i+1, mse_test))
    return mse


def plot_test_mse_vs_n(mse, label=None):
    x = range(1, len(mse)+1)
    plt.plot(x, mse, label=label)
    plt.xlabel('num workers')
    plt.ylabel('Test MSE')

if __name__ == '__main__':
    for i in range(len(expts)):
        mse = increase_n_keep_lr_test_mse(config, expts[i])
        plot_test_mse_vs_n(mse, expts[i]['label'])
    plt.legend()
    plt.show()