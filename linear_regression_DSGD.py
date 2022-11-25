import numpy as np
import pdb
from numpy import linalg as LA
import matplotlib.pyplot as plt
from data_helpers import create_distr_dataset, create_test_dataset, get_data_distr, get_data
from topology import get_diff_matrix, diffuse


def train(config, expt):
    if 'num_clients' in expt.keys(): 
        n_clients = expt['num_clients'] # overwrite config when expt has a different (lower) number of nodes
    else: 
        n_clients = config['num_clients']
    comm_matrix = get_diff_matrix(expt, n_clients)
    if config['log']: print(comm_matrix)

    x = np.zeros((n_clients, config['hidden_dim']))
    A, b, L_i_max, x_nat = get_data_distr(n_clients, log=config['log'], path=config['path'])

    lr = 1/L_i_max * config['lr_factor']
    if config['log']: print('Learning rate: %.5f' % lr)

    for step in range(config['steps']):
        mse = 0
        for i in range(n_clients):
            diff = A[i] @ x[i] - b[i]
            mse += diff @ diff / 2
            grad = A[i].T @ diff
            x[i] -= lr * grad
        mse /= n_clients

        # diffuse params
        x = diffuse(comm_matrix, x, step, expt)

        if config['log'] and step % 20 == 0:
            print('Step %d -- Train MSE: %.3f' % (step, mse))

        if config['mse_th'] is not None and mse < config['mse_th']:
            if config['log']: print('MSE < %.3f at step %d' % (config['mse_th'], step))
            return step
    
    if config['eval']:
        path = path=config['path'] + '_n' + str(config['num_clients']) + '_h' + str(config['hidden_dim'])
        A_test = np.load(path + '/A_test.npy')
        b_test = np.load(path + '/b_test.npy')
        mse = np.sum((A_test @ x.T - b_test[:,None])**2)/(2*n_clients)
        print('Average test MSE: %.5f' % mse)
        # pdb.set_trace()
        # gg = np.zeros(100)
        # for i in range(n_clients):
        #     diff = A[i] @ x[i] - b[i]
        #     grad = A[i].T @ diff
        #     gg += grad
        # pdb.set_trace()
        return mse
    
    return -1

config = {
    'mse_th': None,
    # 'steps': 200,
    'steps': 2000,
    'num_clients': 16,
    'hidden_dim': 100,
    'eval': True,
    'path': 'data/linear_regression_distr',
    'log': True,
    'lr_factor': 2.7
}
# expt = {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 3}
expt = {'topology':'FC_alpha', 'local_steps': 10, 'label': 'FC alpha, 10 local steps'}

if __name__ == '__main__':

    train(config, expt, 16)

    '''
    steps, lr_factors = search_optimal_lr(topology='solo')
    plot_convergence_vs_lr(steps, lr_factors, label='solo', show=False)
    steps, lr_factors = search_optimal_lr(topology='exponential_graph')
    plot_convergence_vs_lr(steps, lr_factors, label='Exponential graph', show=False)
    steps, lr_factors = search_optimal_lr(topology='EG_multi_step')
    plot_convergence_vs_lr(steps, lr_factors, label='multi step EG', show=False)
    steps, lr_factors = search_optimal_lr(topology='ring')
    plot_convergence_vs_lr(steps, lr_factors, label='Ring')
    '''
    
    #train(mse_threshold=0.001, topology='exponential_graph', eval=True, local_steps=0)
    #train(num_clients=16, topology='solo', eval=True, mse_threshold=0.001, lr_factor=1.2)
    #create_distr_dataset(16, iid=False)
    # create_test_dataset()
    #for i in range(1,17):
    #    print(i)
    #    train(num_clients=i, mse_threshold=0.001, topology='fully_connected', eval=True, log=False)

