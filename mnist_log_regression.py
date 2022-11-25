import numpy as np
import pdb
import matplotlib.pyplot as plt
from data_helpers import get_mnist, get_mnist_distr
from topology import get_diff_matrix, diffuse


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1)[:,None]
def batch_softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=2)[:, :, None]

def eval(A_test, b_test, X):
    ''' return test loss and accuracy '''
    if len(X.shape) == 2:   # centalized
        pred = softmax(A_test @ X)
        b_hat = np.argmax(pred, axis=1)
        acc = np.sum(b_hat == b_test) / b_test.shape[0]
        loss = - np.mean(np.log(pred[np.arange(b_test.shape[0]), b_test])) 
    else:                   # decentralized
        pred = batch_softmax(A_test @ X)
        b_hat = np.argmax(pred, axis=2)
        acc = np.sum(b_hat == b_test) / (b_test.shape[0]*X.shape[0])
        loss = - np.mean(np.log(pred[:, np.arange(b_test.shape[0]), b_test]))
    return acc, loss

def plot_heatmaps(X):
    plt.subplots(2,5, figsize=(24,10))
    for i in range(10):
        l1 = plt.subplot(2, 5, i + 1)
        l1.imshow(X[:784, i].reshape(28, 28), interpolation='nearest',cmap=plt.cm.RdBu)
        l1.set_xticks(())
        l1.set_yticks(())
        l1.set_xlabel('Class %i' % i)
    plt.suptitle('Image of the 784 weights for each 10 trained classifiers')
    plt.show()


def train_mnist_centralized(config, expt, do_plot=False):
    ''' train centralized instead of simulating fully connected, which is equivalent'''

    A, b, A_test, b_test = get_mnist()
    b_one_hot = np.zeros((b.shape[0], 10))
    b_one_hot[np.arange(b.shape[0]), b] = 1  

    X = np.zeros((785, 10))
    base_step_size = 0.5
    step_size = base_step_size * config['lr_factor']
    batch_size = config['batch_size'] * config['n_nodes']
    n_batch = int(60000 / (batch_size))

    accuracies = [] 
    train_losses = []
    for epoch in range(config['epochs']):

        batch_perm = np.random.permutation(np.arange(60000))
    
        for batch in range(n_batch):
            samples = batch_perm[batch*batch_size : (batch+1)*batch_size]
            pred = softmax(A[samples] @ X)
            grad = A[samples].T @ (pred - b_one_hot[samples]) / A[samples].shape[0]
            if 'lambda' in expt.keys():
                grad += expt['lambda'] * X
            X -= step_size * grad 
            
            train_loss = - np.mean(np.log(pred[np.arange(b[samples].shape[0]), b[samples]] + 1e-10))
            if 'lambda' in expt.keys():
                train_loss += expt['lambda']/2 * np.mean(X**2)
            train_losses.append(train_loss)
            if 'train_loss_th' in config.keys() and train_loss < config['train_loss_th']:
                return accuracies, None, train_losses

        # evaluate
        acc, test_loss = eval(A_test, b_test, X)
        accuracies.append(acc)
        if config['log']: print('Epoch % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss %.3f' % (epoch, float(acc*100), test_loss, train_loss))      

        # stop if threhsold reached
        if 'acc_th' in config.keys() and config['acc_th'] < acc:
            return accuracies, test_loss, train_losses
        if 'loss_th' in config.keys() and test_loss < config['loss_th']:
            return accuracies, test_loss, train_losses

    if do_plot:
        plot_heatmaps(X)

    return accuracies, test_loss, train_losses

def train_mnist(config, expt):
    if expt['topology'] == 'centralized':
        return train_mnist_centralized(config, expt)

    if 'n_nodes' in expt.keys(): 
        n_nodes = expt['n_nodes']
    else: 
        n_nodes = config['n_nodes']

    comm_matrix = get_diff_matrix(expt, n_nodes)
    if config['log']: print(comm_matrix)

    A, b, b_OH, A_test, b_test = get_mnist_distr(n_nodes, config['data_distr'])
    samples_per_node = np.array(A).shape[1]
    n_batch = int(samples_per_node // config['batch_size'])
    
    X = np.zeros((n_nodes, 785, 10))
    base_step_size = 0.5
    step_size = base_step_size * config['lr_factor']

    step = 0
    accuracies = []
    train_losses = []
    for epoch in range(config['epochs']):
        batch_perm = np.random.permutation(np.arange(samples_per_node))
    
        for batch in range(n_batch):
            train_loss = 0
            samples = batch_perm[batch*config['batch_size'] : (batch+1)*config['batch_size']]
            for i in range(n_nodes):
                pred = softmax(A[i][samples] @ X[i])
                grad = A[i][samples].T @ (pred - b_OH[i][samples]) / A[i][samples].shape[0]
                if 'lambda' in expt.keys():
                    grad += expt['lambda'] * X[i]
                X[i] -= step_size * grad
                
                train_loss += - np.mean(np.log(pred[np.arange(b[i][samples].shape[0]), b[i][samples]]))
                if 'lambda' in expt.keys():
                    train_loss += expt['lambda']/2 * np.mean(X[i]**2)
            train_loss /= n_nodes
            train_losses.append(train_loss)
            if 'train_loss_th' in config.keys() and train_loss < config['train_loss_th']:
                return accuracies, None, train_losses
            # diffuse params
            X = diffuse(comm_matrix, X, step, expt)
            step += 1

        # evaluate
        acc, test_loss = eval(A_test, b_test, X)
        accuracies.append(acc)
        if config['log']: print('Epoch % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss %.3f' % (epoch, float(acc*100), test_loss, train_loss))      

        # stop if threhsold reached
        if 'acc_th' in config.keys() and config['acc_th'] < acc:
            return accuracies, test_loss, train_losses
        if 'loss_th' in config.keys() and test_loss < config['loss_th']:
            return accuracies, test_loss, train_losses

    return accuracies, test_loss, train_losses

config = {
    'loss_th': 0.2,
    'data_distr': 'iid',     # 'iid' or 'non-iid'
    'epochs': 10,
    # 'batch_size': 128,
    # 'n_nodes': 100,
    'batch_size': 32,
    'n_nodes': 32,
    'eval': True,
    'path': 'data/mnist',
    'log': True,
    'lr_factor': 4,
}
# expt = {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 0, 'lambda': 1e-1}
# expt = {'topology': 'fully_connected', 'local_steps': 58}
# expt = {'topology': 'fully_connected', 'local_steps': 58, 'lambda': 1e-4}
# expt = {'topology': 'random', 'degree': 7, 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'exponential_graph', 'local_steps': 0}


if __name__ == '__main__':
    train_mnist(config, expt)
    # train_mnist_centralized()
