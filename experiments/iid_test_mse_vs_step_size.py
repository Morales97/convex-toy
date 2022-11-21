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
    # 'steps': 2000,
    'num_clients': 16,
    'hidden_dim': 100,
    'eval': True,
    'path': 'data/linear_regression_distr',
    'log': False,
    'lr_factors': np.linspace(0.1, 4, 30),
    # 'lr_factors': np.linspace(0.1, 2, 30),
}

expts_base = {'local_steps': 0}
expts = [
    # {'topology': 'fully_connected', 'label': 'Fully connected', **expts_base},
    # {'topology': 'exponential_graph', 'label': 'Exponential Graph', **expts_base},
    #{'topology': 'EG_time_varying', 'label': 'time varying EG', **expts_base},
    #{'topology': 'EG_multi_step', 'label': 'multi step EG', **expts_base},
    # {'topology': 'ring', 'label': 'Ring', **expts_base},
    {'topology':'fully_connected', 'local_steps': 10, 'label': 'Fully connected, 10 local steps'},
    {'topology':'FC_randomized_local_steps', 'local_steps': 10, 'label': 'FC randomized, 10 local steps'},
    {'topology':'FC_alpha', 'local_steps': 10, 'label': 'FC alpha, 10 local steps'},
]

def sweep_step_sizes(config, expt):
    mse = []
    for lr_factor in config['lr_factors']:
        config['lr_factor'] = lr_factor
        mse_test = train(config, expt, config['num_clients'])
        if mse_test > 0.1:
            break 
        mse.append(mse_test)
    return mse


def increase_n_keep_lr_test_mse(config, expt):
    mse = []
    for i in range(config['num_clients']):
        mse_test = train(config, expt, n_clients=i+1)
        mse += [mse_test]
        print('Workers: %d\tTest MSE: %.5f'%(i+1, mse_test))
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