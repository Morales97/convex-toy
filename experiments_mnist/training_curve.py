from xml.parsers.expat import model
import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from mnist_log_regression import train_mnist



def plot_test_accuracy_vs_epoch(config, expt, accuracies):
    epochs = np.arange(config['epochs'])
    assert len(epochs) == len(accuracies)
    
    plt.plot(epochs, accuracies, label=expt['label'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')

def plot_train_loss_vs_epoch(config, expt, train_loss):
    epochs = np.arange(config['epochs'])
    # steps = np.arange(config['epochs'] * int(60000 // (config['n_nodes']*config['batch_size'])))
    # assert len(steps) == len(train_loss)
    train_loss = np.array(train_loss)
    train_loss = train_loss.reshape(-1, int(60000 // (config['n_nodes']*config['batch_size'])))
    train_loss_avg = np.mean(train_loss, axis=1)
    assert len(epochs) == len(train_loss_avg)
    # plt.plot(steps, train_loss, label=expt['label'])
    plt.plot(epochs, train_loss_avg, label=expt['label'])
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')

def accuracy_vs_epoch(config, expts):
    for i in range(len(expts)):
        accuracies, loss_test, loss_train = train_mnist(config, expts[i])
        plot_test_accuracy_vs_epoch(config, expts[i], accuracies)
        # plot_train_loss_vs_epoch(config, expts[i], loss_train)
    plt.legend()
    plt.show()

def acc_and_loss_vs_epoch(config, expts):
    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    for i in range(len(expts)):
        new_config = {**config, **expts[i]} # NOTE join the two dicts, such that values in config can be OVERWRITTEN with values in expts. Need to explicite use the value in each expts entry
        accuracies, loss_test, loss_train, _, _ = train_mnist(new_config, expts[i])
        epochs = np.arange(new_config['epochs'])
        assert len(epochs) == len(accuracies)
        axes[0].plot(epochs, accuracies, label=expts[i]['label'])
        axes[1].plot(epochs, loss_test, label=expts[i]['label'])
    plt.legend()
    plt.show()

def node_disagreement(config, expts):
    for i in range(len(expts)):
        new_config = {**config, **expts[i]} # NOTE join the two dicts, such that values in config can be OVERWRITTEN with values in expts. Need to explicite use the value in each expts entry
        accuracies, loss_test, loss_train, _, model_vars = train_mnist(new_config, expts[i])
        steps = np.arange(len(model_vars))
        plt.plot(steps, model_vars, label=expts[i]['label'])
    plt.xlabel('Iterations')
    plt.ylabel('Model diff norm')
    plt.legend()
    plt.show()

config = {
    'data_distr': 'iid',     # 'iid' or 'non-iid'
    'epochs': 20,
    'steps': 58*20,
    'batch_size': 32,
    # 'batch_size': 10,
    'n_nodes': 32,
    # 'n_nodes': 500*5,
    'steps_eval': 58,
    'eval': True,
    'path': 'data/mnist',
    'log': True,
    'lr_factor': 1,
    # 'lr_factor': 4,
    'net': 'log_reg',
}
expts = [
    # {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'FC, 5 local steps', 'local_steps': 5},
    # {'topology': 'solo', 'label': 'solo', 'local_steps': 0},
    # {'topology': 'centralized', 'label': 'Fully connected, lmbda 1e-3', 'local_steps': 0, 'lambda': 1e-3},
    # {'topology': 'centralized', 'label': 'Fully connected, lmbda 1e-4', 'local_steps': 0, 'lambda': 1e-4},
    # {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 4 local steps', 'local_steps': 4},
    {'topology': 'fully_connected', 'label': 'Fully connected, all local steps', 'local_steps': 58*20},
    {'topology': 'fully_connected', 'label': 'Fully connected, all local steps - no split (IID)', 'local_steps': 58*20, 'data_split': 'no'},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 290 local steps', 'local_steps': 290},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 580 local steps', 'local_steps': 580},
    # {'topology': 'fully_connected', 'label': 'FC, 58 steps, lr_f 0.1', 'local_steps': 58, 'lr_factor': 0.1},
    # {'topology': 'fully_connected', 'label': 'FC, 58 steps, lr_f 0.5', 'local_steps': 58, 'lr_factor': 0.5},
    # {'topology': 'fully_connected', 'label': 'FC, 58 steps, lr_f 1', 'local_steps': 58, 'lr_factor': 1},
    # {'topology': 'fully_connected', 'label': 'FC, 58 steps, lr_f 2', 'local_steps': 58, 'lr_factor': 2},
    # {'topology': 'fully_connected', 'label': 'FC, 58 steps, lr_f 4', 'local_steps': 58, 'lr_factor': 4},

    # {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    # {'topology': 'EG_time_varying', 'label': 'EG time-varying', 'local_steps': 0},
    # {'topology': 'EG_time_varying_random', 'label': 'EG time-varying random', 'local_steps': 0},
    # {'topology': 'EG_multi_step', 'label': 'EG multi-step', 'local_steps': 0},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph, 58 local steps', 'local_steps': 58},
    # {'topology': 'random', 'degree': 3, 'label': 'random (degree: 3)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 5, 'label': 'random (degree: 5)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 9, 'label': 'random (degree: 9)', 'local_steps': 0},
    # {'topology': 'ring', 'label': 'ring', 'local_steps': 0},

]



if __name__ == '__main__':
    # acc_and_loss_vs_epoch(config, expts)
    node_disagreement(config, expts)