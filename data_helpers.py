
import os
import numpy as np
from numpy import linalg as LA
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data

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

def get_mnist_distr(n_nodes = 100, data_distr = 'iid'):
    X_train, y_train, X_test, y_test = get_mnist()
    assert data_distr in ['iid', 'non-iid']


    if data_distr == 'iid':
        indxs = np.arange(60000)
    if data_distr == 'non-iid':
        indices = []
        for i in range(10):
            indices.append(np.where(y_train == i)[0])
        indxs = np.concatenate(indices)     # sorted indices by class

    n_shards = n_nodes * 2 # assign 2 random shards to each node
    n_samples_per_shard = 60000 // n_shards
    random_shard_perm = np.random.permutation(np.arange(n_shards))
    print('Samples per node: %d' % int(n_samples_per_shard*2))     
    
    indices_per_node = []
    for i in range(n_nodes):
        shard_1 = random_shard_perm[2*i]
        shard_2 = random_shard_perm[2*i+1]
        samples_shard_1 = indxs[shard_1*n_samples_per_shard : (shard_1+1)*n_samples_per_shard]
        samples_shard_2 = indxs[shard_2*n_samples_per_shard : (shard_2+1)*n_samples_per_shard]
        indices_per_node.append(np.concatenate((samples_shard_1, samples_shard_2)))

    X_distr = []
    y_distr = []
    y_distr_OH = []
    for i in range(n_nodes):    
        X_distr += [X_train[indices_per_node[i]]]
        y_distr += [y_train[indices_per_node[i]]]

        y_one_hot = np.zeros((y_distr[-1].shape[0], 10))
        y_one_hot[np.arange(y_distr[-1].shape[0]), y_distr[-1]] = 1  
        y_distr_OH += [y_one_hot]

    return X_distr, y_distr, y_distr_OH, X_test, y_test


def get_mnist_distr_full_data(n_nodes = 100, n_samples_per_node=60000):
    ''' Every node has access to the full dataset'''
    X_train, y_train, X_test, y_test = get_mnist()
    X_train = X_train[:n_samples_per_node]
    y_train = y_train[:n_samples_per_node]

    X_distr = []
    y_distr = []
    y_distr_OH = []
    for i in range(n_nodes):    
        X_distr += [X_train]
        y_distr += [y_train]

        y_one_hot = np.zeros((y_distr[-1].shape[0], 10))
        y_one_hot[np.arange(y_distr[-1].shape[0]), y_distr[-1]] = 1  
        y_distr_OH += [y_one_hot]

    return X_distr, y_distr, y_distr_OH, X_test, y_test


def get_mnist_pytorch_iid(config):
    '''
    Return the full dataset, random sampling with replacement
    '''
    traindata = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
    sampler = data.RandomSampler(traindata, replacement=True, num_samples=config['batch_size'])   # NOTE I think num_samples is the total amount of samples to be sampled
    train_loader = data.DataLoader(traindata, sampler=sampler, batch_size=config['batch_size'])

    test_loader = data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            ), batch_size=10*config['batch_size'], shuffle=True)

    return train_loader, test_loader

def get_mnist_pytorch_split(config):
    '''
    Split dataset randomly between workers -> breaks IID
    '''
    traindata = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
    
    # sampler = data.RandomSampler(traindata, replacement=True, num_samples=config['batch_size'])   
    traindata_split = data.random_split(traindata, [int(traindata.data.shape[0] / config['n_nodes']) for _ in range(config['n_nodes'])])
    # train_loader = [data.DataLoader(x, batch_size=config['batch_size'], sampler=sampler) for x in traindata_split]
    train_loader = [data.DataLoader(x, batch_size=config['batch_size'], shuffle=True) for x in traindata_split]

    test_loader = data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            ), batch_size=10*config['batch_size'], shuffle=True)

    return train_loader, test_loader

def get_mnist_pytorch(config):
    if config['data_split'] == 'yes':
        return get_mnist_pytorch_split(config)
    elif config['data_split'] == 'no':
        return get_mnist_pytorch_iid(config)
    else:
        raise Exception('data split modality not supported')

def get_next_batch(config, train_loader, i):
    '''
    Sample a batch of MNIST samples. Supports data split or sampling from full dataset
    '''
    if config['data_split'] == 'yes':
        pdb.set_trace()
        input, target = next(iter(train_loader[i]))
    elif config['data_split'] == 'no':
        pdb.set_trace()
        input, target = next(iter(train_loader))

    return input, target