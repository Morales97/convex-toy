from cmath import exp
import numpy as np
import pdb

def get_diff_matrix(expt, num_clients):
    topology = expt['topology']

    if topology == 'solo' or num_clients == 1:
        W = np.eye(num_clients)

    elif topology == 'ring':
        if num_clients == 2:
            W = np.ones((2,2))/2
        else:
            W = 1/3 * (np.eye(num_clients) + np.eye(num_clients, k=1) + np.eye(num_clients, k=-1) + np.eye(num_clients, k=num_clients-1) + np.eye(num_clients, k=-num_clients+1)) 
    
    elif topology in ['fully_connected', 'FC_randomized_local_steps']:
        W = np.ones((num_clients, num_clients)) / num_clients

    elif topology == 'FC_alpha':    #
        alpha = expt['local_steps'] / (1+expt['local_steps'])
        W = alpha * np.eye(num_clients) + (1-alpha) * np.ones((num_clients,num_clients)) / num_clients
    
    elif topology == 'exponential_graph':
        connections = [2**i for i in range(int(np.log2(num_clients)))]
        W = np.eye(num_clients)/(len(connections)+1)
        for c in connections:
            W += (np.eye(num_clients, k=c) + np.eye(num_clients, k=c-num_clients))/(len(connections)+1)
    
    elif topology in ['EG_time_varying', 'EG_multi_step']:
        connections = [2**i for i in range(int(np.log2(num_clients)))]
        W = [np.eye(num_clients)/2 for _ in connections]
        for i, c in enumerate(connections):
            W[i] += (np.eye(num_clients, k=c) + np.eye(num_clients, k=c-num_clients))/2

    return W

def diffuse(W, x, step, expt):
    if expt['topology'] == 'FC_alpha':          # implicit local steps with W
        return W @ x

    if expt['topology'] == 'FC_randomized_local_steps':     # take a local step with prob (1-p)
        p = 1 / (1+expt['local_steps'])
        if np.random.uniform() < p:
            return W @ x
        else:
            return x

    if expt['local_steps'] > 0 and step % (expt['local_steps']+1) != 0:
        return x

    if isinstance(W, list):
        if 'time_varying' in expt['topology']:      # time-varying
            x = W[step%len(W)] @ x
        else:
            for i in range(len(W)):         # multi-step
                x = W[i] @ x
    else:
        x = W @ x
    return x