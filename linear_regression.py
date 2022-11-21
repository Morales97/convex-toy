import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import os


def get_data(create_dataset=False, n_samples=400, hidden_dim=100, path='data/linear_regression_h'):

    path = path + str(hidden_dim) + '_0'

    if create_dataset:
        if not os.path.exists(path):
            os.makedirs(path)
        x_natural = np.random.normal(size=hidden_dim)
        A = np.random.normal(size=(400, hidden_dim))
        A_test = np.random.normal(size=(400, hidden_dim))
        b = A @ x_natural
        b_test = A_test @ x_natural

        np.save(path + '/x_natural.npy', x_natural)
        np.save(path + '/A_train.npy', A)
        np.save(path + '/A_test.npy', A_test)
        np.save(path + '/b_train.npy', b)
        np.save(path + '/b_test.npy', b_test)

    else: 
        x_natural = np.load(path + '/x_natural.npy')
        A = np.load(path + '/A_train.npy')
        A_test = np.load(path + '/A_test.npy')
        b = np.load(path + '/b_train.npy')
        b_test = np.load(path + '/b_test.npy')

        #training, testing = np.load('data/logistic_regression/training.npz'), np.load('data/logistic_regression/testing.npz')
        #A, b = training['A'], training['b']             # (546, 10), (546, )
        #A_test, b_test = testing['A'], testing['b']     # (137, 10), (137, )   
    
    return x_natural, A, b, A_test, b_test



def get_data2(hidden_dim=100, path='data/linear_regression_distr_n16', node=0):
    path = path + '_h' + str(hidden_dim)

    x_natural = np.load(path + '/x_natural.npy')
    A = np.load(path + '/A_train_'+str(node)+'.npy')
    b = np.load(path + '/b_train_'+str(node)+'.npy')

    return x_natural, A, b, None, None



def get_convergence_vs_lr(hidden_dim=100, mse_threshold=0.01):
    x_natural, A, b, A_test, b_test = get_data(hidden_dim=hidden_dim)
    
    L = LA.svd(A.T @ A)[1][0]     
    print('Lipschitz constant: %.3f\t1/L: %.5f' % (L, 1/L))

    #learning_rates = [np.linspace(1e-5, 1/L, num=5)] + [np.linspace(1/L, 2/L, num=5)][1:]
    learning_rates = [0.01/L, 0.05/L, 0.08/L, 1/L, 1.2/L, 1.5/L, 1.8/L, 2/L]
    conv_steps = []
    
    for lr in learning_rates:
        conv_steps += [train(A, b, lr, hidden_dim, log=False, mse_threshold=mse_threshold)]
        print('Step size: %.5f\tSteps until mse < %.2f: %d' % (lr, mse_threshold, conv_steps[-1]))



def train(A, b, lr, hidden_dim, log=True, mse_threshold=None):

    # init model
    x = np.zeros(hidden_dim)
    
    # train
    steps = 1000
    for step in range(steps):

        diff = A @ x - b
        mse = diff.T @ diff / 2
        grad = A.T @ diff   
        x -= lr * grad

        if log and step % 100 == 0:
            print('Step %d --- train MSE: %.4f' % (step, mse))
        
        if mse_threshold is not None and mse < mse_threshold:
            if log: print('MSE < %.2f at step %d' % (mse_threshold, step))
            return step

    if mse_threshold is not None:
        return -1
    return x


def main(lr=None, hidden_dim=100, mse_threshold=None):

    #x_natural, A, b, A_test, b_test = get_data(hidden_dim=hidden_dim, create_dataset=True)
    x_natural, A, b, A_test, b_test = get_data2(hidden_dim=hidden_dim)
    
    L = LA.svd(A.T @ A)[1][0]     # Spectral norm is largest eigenvalue of Hessian
    print('Lipschitz constant: %.3f' % L)
    
    if lr is None: lr = 1/L
    print('Learning rate: %.4f' % lr)

    x = train(A, b, lr, hidden_dim, mse_threshold=mse_threshold)

    # evaluate
    if mse_threshold is None:
        diff = A_test @ x - b_test
        mse = diff.T @ diff / 2
        print('test MSE: %.2f' % (mse))
        dist = LA.norm(x_natural - x)**2
        print('Distance to x_natural: %.4f' % (dist))


if __name__ == "__main__":
    
    #get_convergence_vs_lr()
    #main()
    main(mse_threshold=0.01)
    #create_dataset()