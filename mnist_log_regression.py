import numpy as np
import pdb
import matplotlib.pyplot as plt
from data_helpers import get_mnist
from topology import get_diff_matrix, diffuse


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1)[:,None]
    
def accuracy(A_test, b_test, X):
    pred = A_test @ X
    b_hat = np.argmax(pred, axis=1)
    acc = np.sum(b_hat == b_test) / b_test.shape[0]
    return acc

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

def train_mnist_centralized():

    A, b, A_test, b_test = get_mnist()
    b_one_hot = np.zeros((b.shape[0], np.max(b)+1))
    b_one_hot[np.arange(b.shape[0]),b] = 1  
    
    X = np.zeros((785, 10))
    step_size = 0.5

    for step in range(200):
        pred = softmax(A @ X)
        
        grad = A.T @ (pred - b_one_hot)
        X -= step_size * grad / A.shape[0]
        
        if step % 50 == 0:
            loss_class_c = np.log(pred[np.arange(b.shape[0]), b])               # class c is 1
            loss_classes_j = np.sum(np.log((1-pred*(1-b_one_hot))), axis=1)     # classes j are 0
            loss = - np.sum(loss_class_c + loss_classes_j)/A.shape[0]
            print(loss)
    
    acc = accuracy(A_test, b_test, X)
    print('Test accuracy: %.2f' % float(acc*100))

    plot_heatmaps(X)


def train_mnist(config, expt, n_clients):
    comm_matrix = get_diff_matrix(expt, n_clients)
    if config['log']: print(comm_matrix)

    A, b, A_test, b_test = get_mnist()
    b_one_hot = np.zeros((b.shape[0], np.max(b)+1))
    b_one_hot[np.arange(b.shape[0]),b] = 1  
    
    X = np.zeros((785, 10))
    step_size = 0.1

    for step in range(200):
        pred = softmax(A @ X)
        
        grad = A.T @ (pred - b_one_hot) / A.shape[0]
        X -= step_size * grad
        
        if step % 50 == 0:
            loss_class_c = np.log(pred[np.arange(b.shape[0]), b])               # class c is 1
            loss_classes_j = np.sum(np.log((1-pred*(1-b_one_hot))), axis=1)     # classes j are 0
            loss = - np.sum(loss_class_c + loss_classes_j)/A.shape[0]
            print(loss)
    
    acc = accuracy(A_test, b_test, X)
    print('Test accuracy: %.2f' % float(acc*100))





if __name__ == '__main__':
    train_mnist_centralized()
