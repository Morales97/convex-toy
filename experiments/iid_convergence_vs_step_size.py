import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt

sys.path.insert(0,  os.path.join(sys.path[0], '..'))
from linear_regression_DSGD import train

config = {
    'mse_th': 0.001,
    'steps': 200,
    'num_clients': 16,
    'hidden_dim': 100,
    'eval': False,
    'path': 'data/linear_regression_distr',
    'log': False,

    'lr_factors': np.linspace(0.5, 5, 23),
}

expts = [
    {'topology':'fully_connected', 'local_steps': 0, 'label': 'Fully connected'},
    {'topology':'fully_connected', 'local_steps': 3, 'label': 'Fully connected, 3 local steps'},
    {'topology':'fully_connected', 'local_steps': 10, 'label': 'Fully connected, 10 local steps'},
    {'topology':'fully_connected', 'local_steps': 1 , 'label': 'Fully connected, 1 local steps'},
    {'topology':'fully_connected', 'local_steps': 20, 'label': 'Fully connected, 20 local steps'},
    # {'topology':'FC_alpha', 'local_steps': 0, 'label': 'Fully connected'},
    # {'topology':'FC_alpha', 'local_steps': 3, 'label': 'Fully connected, 3 local steps'},
    # {'topology':'FC_alpha', 'local_steps': 10, 'label': 'Fully connected, 10 local steps'},
    # {'topology':'FC_alpha', 'local_steps': 1 , 'label': 'Fully connected, 1 local steps'},
    # {'topology':'FC_alpha', 'local_steps': 20, 'label': 'Fully connected, 20 local steps'},
    # {'topology':'FC_randomized_local_steps', 'local_steps': 3, 'label': 'FC randomized, 3 local steps'},
    # {'topology':'FC_alpha', 'local_steps': 3, 'label': 'FC alpha, 3 local steps'},
]

def sweep_step_sizes(config, expt):
    steps = []
    for lr_factor in config['lr_factors']:
        config['lr_factor'] = lr_factor
        step = train(config, expt)
        if step == -1:
            break 
        steps.append(step)
    return steps

def plot_convergence_vs_lr(steps, lr_factors, label=None):
    x = lr_factors[:len(steps)]
    plt.plot(x, steps, label=label)
    plt.xlabel('LR factor (1/L * factor)')
    plt.ylabel('steps until conv')

def get_optimal_step_size(config, expt):
    steps = sweep_step_sizes(config, expt)
    id_min = np.argmin(np.array(steps))     # find faster convergence
    step_size_opt = config['lr_factors'][id_min]    
    alpha = expt['local_steps'] / (1+expt['local_steps'])
    return step_size_opt, alpha

def plot_optimal_step_size():

    expt_solo = {'topology':'solo', 'local_steps': 100000}
    step_size_alpha1, _ = get_optimal_step_size(config, expt_solo)
    step_size_alpha0, alpha = get_optimal_step_size(config, expts[0])
    assert alpha == 0, 'need reference without local step sizes'
    lr_opts = [1]
    alpha_array = [alpha]
    for i in range(len(expts)):
        step_size_opt, alpha = get_optimal_step_size(config, expts[i])
        lr_opts.append(step_size_opt/step_size_alpha0)
        alpha_array.append(alpha)

    alphas = np.linspace(0, 1, 100)
    # y = (1-alphas**2)**(2/3)
    # y = np.maximum((1-alphas**2)**(2/3) * (1-alphas) + alphas * step_size_alpha1/step_size_alpha0, np.ones(len(alphas)) * step_size_alpha1/step_size_alpha0)
    y = np.maximum((1-alphas**2)**(2/3), np.ones(len(alphas)) * step_size_alpha1/step_size_alpha0)
    # delta = (1-alphas**2)
    # y = delta / (delta + delta **(1/3) + 1e-8) 
    # y = (1-alphas**2)**(2/3) + (1-(1-alphas**2)**(2/3)) * step_size_alpha1/step_size_alpha0
    plt.plot(alphas, y, ':', color='red')
    plt.scatter(alpha_array, lr_opts)
    plt.xlabel('alpha')
    plt.ylabel(r'$(1-\alpha^2)^{2/3}$')
    plt.show()

if __name__ == '__main__':
    plot_optimal_step_size()
    # for i in range(len(expts)):
    #     steps = sweep_step_sizes(config, expts[i])
    #     plot_convergence_vs_lr(steps, config['lr_factors'], expts[i]['label'])
    # plt.legend()
    # plt.show()