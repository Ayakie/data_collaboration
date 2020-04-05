# sorce: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/sampling.py

import numpy as np
from keras.datasets import mnist

(X_train, label_train), (X_test, label_test) = mnist.load_data()

num_users = 10
ndat = 100

# To do: add method of gan
def make_anchors(X_train, nanc, anc_type='random'):
    '''
    Parameters
    --------------
    nanc: number of anchor data
    X_train: training data to get min(max) value of feature
    anc_type: 'random', 'gan'
    
    '''
    if anc_type == 'random':
        
        Xanc = np.random.uniform(low=np.min(X_train), high=np.max(X_train), size=(nanc, X_train.shape[1]))
    
    return Xanc

def mnist_iid(X_train, num_users, ndat):
    '''Sample I.I.D user data from MNIST dataset

    Return
    --------
    list of user index(key) and ndat samples index(value) 
    '''

    list_users, all_idx = [], [i for i in range(len(X_train))]
    for i in range(num_users):
        user_idx = set(np.random.choice(all_idx, size=ndat, replace=False))
        all_idx = list(set(all_idx) - user_idx)
        list_users.append(list(user_idx))
    
    return list_users

def mnist_noniid(X_train, label_train,  num_users, ndat, nlabel):
    '''Sample non-I.I.D user data from MNIST dataset
    Parameters
    ----------
    nlabel: int(1~9)
        number of different types of labels which each user has

    Return
    -------
    list of ndat samples index of each user
    '''
    num_shards, num_imgs = nlabel*num_users, int(ndat//nlabel)
    idx_shard = [i for i in range(num_shards)]
    
    total_size = num_users * ndat
    idxs = np.arange(total_size)

    # sort labels
    idxs_labels = np.vstack((idxs, label_train[:total_size]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign nlabel number of label per user
    # dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    list_users = []
    for i in range(num_users):
        user_idx = []
        rand_set = set(np.random.choice(idx_shard, nlabel, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            user_idx = np.hstack((user_idx, idxs[rand*num_imgs:(rand+1)*num_imgs]))
            user_idx = user_idx.astype(int).tolist() # convert dtype from float to int
        list_users.append(user_idx)

    return list_users