import numpy as np
import pdb
import matplotlib.pyplot as plt
from data_helpers import get_mnist, get_mnist_distr, get_mnist_distr_full_data
from topology import get_diff_matrix, diffuse
from plot_helpers import plot_calibration_histogram, plot_heatmaps

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

def eval_soft_acc(A_test, b_test, X):
    if len(X.shape) == 2:   # centalized
        pred = softmax(A_test @ X)
        b_hat = np.argmax(pred, axis=1)
        acc = np.sum(b_hat == b_test) / b_test.shape[0]
        loss = - np.mean(np.log(pred[np.arange(b_test.shape[0]), b_test])) 
        soft_acc = np.sum(pred[np.arange(b_test.shape[0]), b_test]) /  b_test.shape[0]
    else:                   # decentralized
        pred = batch_softmax(A_test @ X)
        b_hat = np.argmax(pred, axis=2)
        acc = np.sum(b_hat == b_test) / (b_test.shape[0]*X.shape[0])
        loss = - np.mean(np.log(pred[:, np.arange(b_test.shape[0]), b_test]))
        soft_acc = np.sum(pred[:, np.arange(b_test.shape[0]), b_test]) / (b_test.shape[0]*X.shape[0])
    return acc, loss, soft_acc


def eval_calibration(A_test, b_test, X, bins=10):
    if len(X.shape) == 2:   # centalized
        pred = softmax(A_test @ X)
        pred_and_label = np.concatenate((np.max(pred, axis=1)[:,None], np.argmax(pred, axis=1)[:,None], b_test[:,None]), axis=1)
        pred_and_label = pred_and_label[pred_and_label[:, 0].argsort()] # (n_samples, 3). contains (0) probability of predicted class (sorted from lower to higher), (1) class prediction and (2) class label 
        pred_and_label = np.split(pred_and_label, bins)
        
        ECE = 0   # Expected Calibration Error
        accuracy_bins = []
        confidence_bins = []
        n_samples = pred.shape[0]
        for i in range(bins):
            acc = np.sum(pred_and_label[i][:,1] == pred_and_label[i][:,2]) / pred_and_label[i].shape[0]
            confidence = np.sum(pred_and_label[i][:,0]) / pred_and_label[i].shape[0]
            accuracy_bins.append(acc)
            confidence_bins.append(confidence)
            ECE += np.abs(acc - confidence) * pred_and_label[i].shape[0] / n_samples
            
        # NOTE I am here
        b_hat = np.argmax(pred, axis=1)
        acc = np.sum(b_hat == b_test) / n_samples
        loss = - np.mean(np.log(pred[np.arange(n_samples), b_test])) 

    else:                   # decentralized
        pred = batch_softmax(A_test @ X)
        
        n_nodes = pred.shape[0]
        n_samples = pred.shape[1]
        avg_ECE = 0
        avg_accuracy_bins = np.zeros(bins)
        avg_confidence_bins = np.zeros(bins)

        for n in range(n_nodes):
            pred_and_label = np.concatenate((np.max(pred[n], axis=1)[:,None], np.argmax(pred[n], axis=1)[:,None], b_test[:,None]), axis=1)
            pred_and_label = pred_and_label[pred_and_label[:, 0].argsort()]
            pred_and_label = np.split(pred_and_label, bins)
            
            ECE_node = 0   # Expected Calibration Error
            for i in range(bins):
                acc = np.sum(pred_and_label[i][:,1] == pred_and_label[i][:,2]) / pred_and_label[i].shape[0]
                confidence = np.sum(pred_and_label[i][:,0]) / pred_and_label[i].shape[0]
                avg_accuracy_bins[i] += acc
                avg_confidence_bins[i] += confidence
                ECE_node += np.abs(acc - confidence) * pred_and_label[i].shape[0] / n_samples
            avg_ECE += ECE_node 
        ECE = avg_ECE / n_nodes
        accuracy_bins = avg_accuracy_bins / n_nodes
        confidence_bins = avg_confidence_bins / n_nodes

        b_hat = np.argmax(pred, axis=2)
        acc = np.sum(b_hat == b_test) / (n_samples*X.shape[0])
        loss = - np.mean(np.log(pred[:, np.arange(n_samples), b_test]))

    return acc, loss, ECE, accuracy_bins, confidence_bins


def train_batch(config, expt, X, A, b, b_OH, samples, step_size):

    # central training
    if expt['topology'] == 'centralized':
        pred = softmax(A[samples] @ X)
        grad = A[samples].T @ (pred - b_OH[samples]) / A[samples].shape[0]
        if 'lambda' in expt.keys():
            grad += expt['lambda'] * X
        X -= step_size * grad 
        
        train_loss = - np.mean(np.log(pred[np.arange(b[samples].shape[0]), b[samples]] + 1e-10))
        if 'lambda' in expt.keys():
            train_loss += expt['lambda']/2 * np.mean(X**2)

    # decentralized training
    else: 
        train_loss = 0
        n_nodes = config['n_nodes']

        # train each worker
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


    return train_loss

def train_batch_random(config, expt, X, A, b, b_OH, samples, step_size):

    # central training
    if expt['topology'] == 'centralized':
        pred = softmax(A[samples] @ X)
        grad = A[samples].T @ (pred - b_OH[samples]) / A[samples].shape[0]
        if 'lambda' in expt.keys():
            grad += expt['lambda'] * X
        X -= step_size * grad 
        
        train_loss = - np.mean(np.log(pred[np.arange(b[samples].shape[0]), b[samples]] + 1e-10))
        if 'lambda' in expt.keys():
            train_loss += expt['lambda']/2 * np.mean(X**2)

    # decentralized training
    else: 
        train_loss = 0
        n_nodes = config['n_nodes']

        # train each worker
        for i in range(n_nodes):
            samples = np.random.choice(np.arange(60000), 32, replace=False)
            pred = softmax(A[i][samples] @ X[i])
            grad = A[i][samples].T @ (pred - b_OH[i][samples]) / A[i][samples].shape[0]
            if 'lambda' in expt.keys():
                grad += expt['lambda'] * X[i]
            X[i] -= step_size * grad
            
            train_loss += - np.mean(np.log(pred[np.arange(b[i][samples].shape[0]), b[i][samples]]))
            if 'lambda' in expt.keys():
                train_loss += expt['lambda']/2 * np.mean(X[i]**2)
        train_loss /= n_nodes

    return train_loss

def train_batch_diag_net(config, expt, X1, X2, A, b, b_OH, samples, step_size):

    # central training
    if expt['topology'] == 'centralized':
        pred = softmax(A[samples] @ np.multiply(X1,X2))
        input_2 = np.einsum('nd,dk->ndk', A[samples], X1)
        grad_2 = np.einsum('ndk,nk->dk', input_2, (pred - b_OH[samples])) / A[samples].shape[0]
        input_1 = np.einsum('nd,dk->ndk', A[samples], X2)
        grad_1 = np.einsum('ndk,nk->dk', input_1, (pred - b_OH[samples])) / A[samples].shape[0]

        X2 -= step_size * grad_2
        X1 -= step_size * grad_1
        
        train_loss = - np.mean(np.log(pred[np.arange(b[samples].shape[0]), b[samples]] + 1e-10))

    else:
        train_loss = 0
        n_nodes = config['n_nodes']

        # train each worker
        for i in range(n_nodes):
            pred = softmax(A[i][samples] @ np.multiply(X1[i],X2[i]))
            input_2 = np.einsum('nd,dk->ndk', A[i][samples], X1[i])
            grad_2 = np.einsum('ndk,nk->dk', input_2, (pred - b_OH[i][samples])) / A[i][samples].shape[0]
            input_1 = np.einsum('nd,dk->ndk', A[i][samples], X2[i])
            grad_1 = np.einsum('ndk,nk->dk', input_1, (pred - b_OH[i][samples])) / A[i][samples].shape[0]

            X2[i] -= step_size * grad_2
            X1[i] -= step_size * grad_1


    return train_loss

def train_batch_two_linear_layers(config, expt, X1, X2, A, b, b_OH, samples, step_size):

    # central training
    if expt['topology'] == 'centralized':
        pred = softmax(A[samples] @ X1 @ X2)
        grad_2 = (A[samples] @ X1).T @ (pred - b_OH[samples]) / A[samples].shape[0]
        pdb.set_trace()
        grad_1 = (X2@b_OH[samples].T) @ A[samples] / A[samples].shape[0]  # the first term
        for k in range(10):
            grad_1 += (A[samples] @ X1).T @ pred[k] / A[samples].shape[0]

        X2 -= step_size * grad_2
        X1 -= step_size * grad_1
        
        train_loss = - np.mean(np.log(pred[np.arange(b[samples].shape[0]), b[samples]] + 1e-10))
        if 'lambda' in expt.keys():
            train_loss += expt['lambda']/2 * np.mean(X**2)


    return train_loss


def train_mnist(config, expt, plt_heatmaps=False, plt_calibration=True):
    
    decentralized = (expt['topology'] != 'centralized') 

    # central training
    if not decentralized:
        A, b, A_test, b_test = get_mnist()
        b_OH = np.zeros((b.shape[0], 10))
        b_OH[np.arange(b.shape[0]), b] = 1  

        X = np.zeros((785, 10))
        # X1 = np.zeros((785, 100))
        # X2 = np.zeros((100, 10))
        X1 = np.random.normal(size=(785, 10))
        X2 = np.random.normal(size=(785, 10))
        batch_size = config['batch_size'] * config['n_nodes']
        samples_per_node = 60000
       
    # decentralized
    else: 
        n_nodes = config['n_nodes']
        comm_matrix = get_diff_matrix(expt, n_nodes)
        if config['log']: print(comm_matrix)

        A, b, b_OH, A_test, b_test = get_mnist_distr(n_nodes, config['data_distr'])
        # A, b, b_OH, A_test, b_test = get_mnist_distr_full_data(n_nodes)
        
        X = np.zeros((n_nodes, 785, 10))
        X1 = np.random.normal(size=(n_nodes, 785, 10))
        X2 = np.random.normal(size=(n_nodes, 785, 10))
        batch_size = config['batch_size']
        samples_per_node = np.array(A).shape[1]

    n_batch = int(samples_per_node // batch_size)
    base_step_size = 0.5
    step_size = base_step_size * config['lr_factor']

    step = 0
    ECE = None  # Expected Calibration Error
    accuracies = []
    train_losses = []
    test_losses = []

    # train looop
    for epoch in range(config['epochs']):
        batch_perm = np.random.permutation(np.arange(samples_per_node))
    
        for batch in range(n_batch):
            samples = batch_perm[batch*batch_size : (batch+1)*batch_size]

            # learning step
            # train_loss = train_batch(config, expt, X, A, b, b_OH, samples, step_size)
            # train_loss = train_batch_two_linear_layers(config, expt, X1, X2, A, b, b_OH, samples, step_size)
            train_loss = train_batch_diag_net(config, expt, X1, X2, A, b, b_OH, samples, step_size)
            # train_loss = train_batch_random(config, expt, X, A, b, b_OH, samples, step_size)
            train_losses.append(train_loss)

            # diffuse params
            if decentralized:
                # if epoch+1 == 20 and batch+1 == n_batch:  # to plot heatmaps of single models
                #     pdb.set_trace()
                X = diffuse(comm_matrix, X, step, expt)
            step += 1

            if step % config['steps_eval'] == 0:
                # X = X1 @ X2
                X = np.multiply(X1, X2)
                acc, test_loss = eval(A_test, b_test, X)
                accuracies.append(acc)
                test_losses.append(test_loss)
                if config['log']: print('Step % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss %.3f' % (step, float(acc*100), test_loss, train_loss))     

        # evaluate
        # acc, test_loss = eval(A_test, b_test, X)
        # acc, test_loss, soft_acc = eval_soft_acc(A_test, b_test, X)
        # acc, test_loss, ECE, accuracy_bins, confidence_bins = eval_calibration(A_test, b_test, X)

        # plot_calibration_histogram(accuracy_bins, confidence_bins)

        # accuracies.append(acc)
        # test_losses.append(test_loss)
        # if config['log']: print('Epoch % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss %.3f' % (epoch, float(acc*100), test_loss, train_loss))      
        # if config['log']: print('Epoch % d -- Test accuracy: %.2f -- Test SOFT accuracy: %.2f -- Test loss: %.3f -- Train loss %.3f' % (epoch, float(acc*100), float(soft_acc*100), test_loss, train_loss))      

        # stop if threhsold reached
        if 'acc_th' in config.keys() and config['acc_th'] < acc:
            if plt_calibration:
                # plot_calibration_histogram(accuracy_bins, confidence_bins)
                pass
            return accuracies, test_loss, train_losses, ECE
        if 'loss_th' in config.keys() and test_loss < config['loss_th']:
            return accuracies, test_loss, train_losses, ECE

    if plt_heatmaps:
        plot_heatmaps(X)
    return accuracies, test_losses, train_losses, ECE


config = {
    'acc_th': 0.915,
    'data_distr': 'iid',     # 'iid' or 'non-iid'
    'epochs': 20,
    'steps_eval': 58,
    'batch_size': 32,
    'n_nodes': 32,
    'eval': True,
    'path': 'data/mnist',
    'log': True,
    'lr_factor': 1,
}
# expt = {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0, 'lambda': 1e-3}
expt = {'topology': 'fully_connected', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 0, 'lambda': 1e-1}
# expt = {'topology': 'fully_connected', 'local_steps': 58}
# expt = {'topology': 'random', 'degree': 7, 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'exponential_graph', 'local_steps': 0}


if __name__ == '__main__':
    train_mnist(config, expt)
    # train_mnist_centralized(config, expt)
