# encoding=utf-8
"""
    Created on 10:38 2019/2/19
    @author: Hangwei Qian
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle as cp
from utils import get_sample_weights

def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    return X

# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(6)[data]
    return YY

def load_data():
    import os
    if os.path.isfile('./data/data_har.npz') == True:
        data = np.load('data/data_har.npz', allow_pickle=True)
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
    else:
        str_folder = './data/' + 'UCI HAR Dataset/'
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

    return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)

def load_data_ucihar():
    import os
    data_dir = './data/'
    saved_filename = 'ucihar_processed.data'
    if os.path.isfile(data_dir + saved_filename) == True:
        print('data is preprocessed in advance! Loading...')
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X_train = data[0][0]
        Y_train = data[0][1]
        X_test = data[1][0]
        Y_test = data[1][1]
    else:
        print('data needs preprocessing first...')
        str_folder = data_dir + 'UCI HAR Dataset/'
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

        obj = [(X_train, Y_train), (X_test, Y_test)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


class data_loader_ucihar(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return self.T(torch.from_numpy(sample).float()), target

    def __len__(self):
        return len(self.samples)


def load_balancedUp(batch_size=64):
    x_train, y_train, x_test, y_test = load_data_ucihar()
    # n_channel should be 9, H: 1, W:128
    x_train, x_test = np.transpose(x_train.reshape((-1, 1, 128, 9)), (0,3,1,2)), np.transpose(x_test.reshape((-1, 1, 128, 9)), (0,3,1,2))

    unique_ytrain, counts_ytrain = np.unique(y_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_ytrain, counts_ytrain)))
    unique_ytest, counts_ytest = np.unique(y_test, return_counts=True)
    print('y_test label distribution: ', dict(zip(unique_ytest, counts_ytest)))
    unique_all, counts_all = np.unique(np.concatenate((y_train, y_test), axis=0), return_counts=True)
    print('y_all label distribution: ', dict(zip(unique_all, counts_all)))

    weights = 100.0 / torch.Tensor(counts_ytrain)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0,0,0,0,0,0,0), std=(1,1,1,1,1,1,1,1,1))
    ])

    train_set = data_loader_ucihar(x_train, y_train, transform)
    test_set = data_loader_ucihar(x_test, y_test, transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print('train_loader batch: ', len(train_loader), 'test_loader batch: ', len(test_loader))
    return train_loader, test_loader
