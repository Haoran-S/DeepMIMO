from __future__ import division
from Data_Generation.function_wmmse_powercontrol import WMMSE_sum_rate
import numpy as np
import argparse
import torch
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--o', default='dataset_deepmimo.pt', help='output file')
parser.add_argument('--num_tasks', default=3, type=int, help='number of tasks')
parser.add_argument('--noise', default=1e-12, type=float)
parser.add_argument('--num_train', default='20000-20000-20000', type=str)
args = parser.parse_args()

tasks_tr = []
tasks_te = []
for t in range(args.num_tasks):
    # Reading input and output sets generated from MATLAB
    filepath = 'Data_Generation/DeepMIMO Dataset/DeepMIMO_dataset_task%d.mat' % (
        t+1)
    f = h5py.File(filepath)
    channel_set_file = {}
    for k, v in f.items():
        channel_set_file[k] = np.transpose(np.array(v))

    channel_set = channel_set_file['H']
    num_samples = channel_set.shape[0]
    num_bs = channel_set.shape[1]
    num_user = channel_set.shape[2]
        
    # compute WMMSE labels
    Pmax = 1
    Pini = np.ones(num_user)
    Y = np.zeros((num_samples, num_user))
    for loop in range(num_samples):
        H = np.reshape(channel_set[loop], (num_user, num_user))
        Y[loop, :] = WMMSE_sum_rate(Pini, H, Pmax, args.noise)
    label_set = Y
    channel_set = channel_set.reshape(num_samples, num_bs*num_user)

    # Parameter initialization
    num_train = int(num_samples -1000)
    num_test = 1000
    np.random.seed(0)
    train_index = np.random.choice(
        range(0, num_samples), size=num_train, replace=False)
    rem_index = set(range(0, num_samples))-set(train_index)
    test_index = list(set(np.random.choice(
        list(rem_index), size=num_test, replace=False)))

    # for t in range(args.n_tasks):
    In_train = channel_set[train_index, :]
    In_test = channel_set[test_index, :]
    Out_train = label_set[train_index, :]
    Out_test = label_set[test_index, :]

    Xtrain = torch.from_numpy(In_train).float()
    Ytrain = torch.from_numpy(Out_train).float()
    tasks_tr.append([t, Xtrain.clone(), Ytrain.clone()])
    print(Xtrain.shape, Ytrain.shape)

    Xtest = torch.from_numpy(In_test).float()
    Ytest = torch.from_numpy(Out_test).float()
    tasks_te.append([t, Xtest.clone(), Ytest.clone()])
    print(Xtest.shape, Ytest.shape)

torch.save([tasks_tr, tasks_te, args], args.o)
