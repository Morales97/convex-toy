
import os
import numpy as np
from numpy import linalg as LA


def create_distr_dataset(num_clients, n_samples=400, hidden_dim=100, path='data/linear_regression_distr_var0.001_n16_h', iid=True):
    '''Create a dataset for distributed training
        can chose either IID (same x_natural) or heterogeneous (sampling additive noise to x_natural)
    ''' 
    path = path + str(hidden_dim)
    if not os.path.exists(path):
        os.makedirs(path)

    x_natural = np.random.normal(size=hidden_dim)
    np.save(path + '/x_natural.npy', x_natural)
    
    # IID case
    if iid:
        for i in range(num_clients):
            A = np.random.normal(size=(n_samples, hidden_dim))
            b = A @ x_natural
            np.save(path + '/A_train_'+str(i)+'.npy', A)
            np.save(path + '/b_train_'+str(i)+'.npy', b)

    # non-IID
    else:
        for i in range(num_clients):
            x_noise = np.random.normal(size=hidden_dim, scale=0.001)
            A = np.random.normal(size=(n_samples, hidden_dim))
            b = A @ (x_natural + x_noise)
            np.save(path + '/A_train_'+str(i)+'.npy', A)
            np.save(path + '/b_train_'+str(i)+'.npy', b)


def create_test_dataset(n_samples=400, hidden_dim=100, path='data/linear_regression_distr_n16_h'):
    '''
    create a test dataset for a given x_natural
    ''' 
    path = path + str(hidden_dim)
    x_natural = np.load(path + '/x_natural.npy')
    
    A = np.random.normal(size=(n_samples, hidden_dim))
    # b = A @ (x_natural + np.random.normal(size=hidden_dim, scale=0.001))
    b = A @ x_natural
    np.save(path + '/A_test.npy', A)
    np.save(path + '/b_test.npy', b)

def get_data(path):

    x_natural = np.load(path + '/x_natural.npy')
    A = np.load(path + '/A_train.npy')
    A_test = np.load(path + '/A_test.npy')
    b = np.load(path + '/b_train.npy')
    b_test = np.load(path + '/b_test.npy')

    return x_natural, A, b, A_test, b_test

def get_data_distr(num_clients, hidden_dim=100, path='data/linear_regression_distr', log=True):

    A_distr = []
    b_distr = []
    L_i_max = 0

    path = path + '_n16' + '_h' + str(hidden_dim)

    for i in range(num_clients):
        A =  np.load(path + '/A_train_'+str(i)+'.npy')
        b =  np.load(path + '/b_train_'+str(i)+'.npy')
        A_distr += [A]
        b_distr += [b]
        L_i = LA.svd(A.T @ A)[1][0]    
        if log: print('Lipschitz of worker %d: %.3f' % (i, L_i))
        if L_i > L_i_max: 
            L_i_max = L_i

    x_nat = np.load(path + '/x_natural.npy')
    return A_distr, b_distr, L_i_max, x_nat

def get_mnist():
    ''' return full mnist dataset '''
    X_train = np.load('data/mnist/X_train.npy')
    X_test = np.load('data/mnist/X_test.npy')
    y_train = np.load('data/mnist/y_train.npy')
    y_test = np.load('data/mnist/y_test.npy')

    return X_train, y_train, X_test, y_test