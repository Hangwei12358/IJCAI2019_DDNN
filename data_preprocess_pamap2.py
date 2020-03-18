# encoding=utf-8
"""
    Created on 10:38 2018/12/17
    @author: Hangwei Qian
    Adapted from: https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
"""
import numpy as np
import torch
import pickle as cp
from torch.utils.data import Dataset, DataLoader
from utils import get_sample_weights, opp_sliding_window


NUM_FEATURES = 52

class data_loader_pamap2(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)


def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.

    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001

    x /= std
    return x


def complete_HR(data):
    """Sampling rate for the heart rate is different from the other sensors. Missing
    measurements are filled

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        HR channel data
    """

    pos_NaN = np.isnan(data)
    idx_NaN = np.where(pos_NaN == False)[0]
    data_no_NaN = data * 0
    for idx in range(idx_NaN.shape[0] - 1):
        data_no_NaN[idx_NaN[idx]: idx_NaN[idx + 1]] = data[idx_NaN[idx]]

    data_no_NaN[idx_NaN[-1]:] = data[idx_NaN[-1]]

    return data_no_NaN


def divide_x_y(data):
    """Segments each sample into time, labels and sensor channels

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Time and labels as arrays, sensor channels as matrix
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]

    return data_t, data_x, data_y

def adjust_idx_labels(data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function adjust the labels picking the labels
    for the protocol settings

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    data_y[data_y == 24] = 0
    data_y[data_y == 12] = 8
    data_y[data_y == 13] = 9
    data_y[data_y == 16] = 10
    data_y[data_y == 17] = 11

    return data_y


def del_labels(data_t, data_x, data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function deletes the nonrelevant labels

    18 ->

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    idy = np.where(data_y == 0)[0]
    labels_delete = idy

    idy = np.where(data_y == 8)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 9)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 10)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 11)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 18)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 19)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 20)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    return np.delete(data_t, labels_delete, 0), np.delete(data_x, labels_delete, 0), np.delete(data_y, labels_delete, 0)


def downsampling(data_t, data_x, data_y):
    """Recordings are downsamplied to 30Hz, as in the Opportunity dataset

    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """

    idx = np.arange(0, data_t.shape[0], 3)

    return data_t[idx], data_x[idx], data_y[idx]


def process_dataset_file(data):
    """Function defined as a pipeline to process individual Pamap2 files

    :param data: numpy integer matrix
        channel data: samples in rows and sensor channels in columns
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into samples-channel measurements (x) and labels (y)
    """

    # Data is divided in time, sensor data and labels
    data_t, data_x, data_y = divide_x_y(data)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # nonrelevant labels are deleted
    data_t, data_x, data_y = del_labels(data_t, data_x, data_y)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # Labels are adjusted
    data_y = adjust_idx_labels(data_y)
    data_y = data_y.astype(int)

    if data_x.shape[0] != 0:
        HR_no_NaN = complete_HR(data_x[:, 0])
        data_x[:, 0] = HR_no_NaN

        data_x[np.isnan(data_x)] = 0

        data_x = normalize(data_x)

    else:
        data_x = data_x
        data_y = data_y
        data_t = data_t

        print("SIZE OF THE SEQUENCE IS CERO")

    data_t, data_x, data_y = downsampling(data_t, data_x, data_y)

    return data_x, data_y


def load_data_pamap2():
    import os
    data_dir = './data/'
    saved_filename = 'pamap2_processed.data'
    if os.path.isfile( data_dir + saved_filename ) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X_train = data[0][0]
        y_train = data[0][1]

        X_validation = data[1][0]
        y_validation = data[1][1]

        X_test = data[2][0]
        y_test = data[2][1]

        return X_train, y_train, X_validation, y_validation, X_test, y_test
    else:
        dataset = 'data/pamap2/'
        # File names of the files defining the PAMAP2 data.
        PAMAP2_DATA_FILES = ['PAMAP2_Dataset/Protocol/subject101.dat',  # 0
                             'PAMAP2_Dataset/Optional/subject101.dat',  # 1
                             'PAMAP2_Dataset/Protocol/subject102.dat',  # 2
                             'PAMAP2_Dataset/Protocol/subject103.dat',  # 3
                             'PAMAP2_Dataset/Protocol/subject104.dat',  # 4
                             'PAMAP2_Dataset/Protocol/subject107.dat',  # 5
                             'PAMAP2_Dataset/Protocol/subject108.dat',  # 6
                             'PAMAP2_Dataset/Optional/subject108.dat',  # 7
                             'PAMAP2_Dataset/Protocol/subject109.dat',  # 8
                             'PAMAP2_Dataset/Optional/subject109.dat',  # 9
                             'PAMAP2_Dataset/Protocol/subject105.dat',  # 10
                             'PAMAP2_Dataset/Optional/subject105.dat',  # 11
                             'PAMAP2_Dataset/Protocol/subject106.dat',  # 12
                             'PAMAP2_Dataset/Optional/subject106.dat',  # 13
                             ]

        X_train = np.empty((0, NUM_FEATURES))
        y_train = np.empty((0))

        X_val = np.empty((0, NUM_FEATURES))
        y_val = np.empty((0))

        X_test = np.empty((0, NUM_FEATURES))
        y_test = np.empty((0))

        counter_files = 0

        print('Processing dataset files ...')
        for filename in PAMAP2_DATA_FILES:
            if counter_files <= 9:
                # Train partition
                try:
                    print('Train... file {0}'.format(filename))
                    data = np.loadtxt(dataset + filename)
                    print('Train... data size {}'.format(data.shape))
                    x, y = process_dataset_file(data)
                    print(x.shape)
                    print(y.shape)
                    X_train = np.vstack((X_train, x))
                    y_train = np.concatenate([y_train, y])
                except KeyError:
                    print('ERROR: Did not find {0} in zip file'.format(filename))

            elif counter_files > 9 and counter_files < 12:
                # Validation partition
                try:
                    print('Val... file {0}'.format(filename))
                    data = np.loadtxt(dataset + filename)
                    print('Val... data size {}'.format(data.shape))
                    x, y = process_dataset_file(data)
                    print(x.shape)
                    print(y.shape)
                    X_val = np.vstack((X_val, x))
                    y_val = np.concatenate([y_val, y])
                except KeyError:
                    print('ERROR: Did not find {0} in zip file'.format(filename))

            else:
                # Testing partition
                try:
                    print('Test... file {0}'.format(filename))
                    data = np.loadtxt(dataset + filename)
                    print('Test... data size {}'.format(data.shape))
                    x, y = process_dataset_file(data)
                    print(x.shape)
                    print(y.shape)
                    X_test = np.vstack((X_test, x))
                    y_test = np.concatenate([y_test, y])
                except KeyError:
                    print('ERROR: Did not find {0} in zip file'.format(filename))

            counter_files += 1

        print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape))

        obj = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
        # f = file(os.path.join(target_filename), 'wb')
        f = open(os.path.join(data_dir+saved_filename), 'wb')

        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
        return X_train, y_train, X_val, y_val, X_test, y_test



def load_fixed_slidwin_balancedUp(dataset="pamap2", batch_size=64, SLIDING_WINDOW_LEN=30, SLIDING_WINDOW_STEP=15):

    if dataset == "pamap2":
        x_train, y_train, x_val, y_val, x_test, y_test = load_data_pamap2()  # (557963, 113), (557963,), (118750, 113), (118750, )

        x_train_win, y_train_win = opp_sliding_window(x_train, y_train, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        x_val_win, y_val_win = opp_sliding_window(x_val, y_val, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        x_test_win, y_test_win = opp_sliding_window(x_test, y_test, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

        unique_ytrain, counts_ytrain = np.unique(y_train_win, return_counts=True)

        weights = 100.0/ torch.Tensor(counts_ytrain)
        weights = weights.double()
        sample_weights = get_sample_weights(y_train_win, weights)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_set = data_loader_pamap2(x_train_win, y_train_win)
        val_set = data_loader_pamap2(x_val_win, y_val_win)
        test_set = data_loader_pamap2(x_test_win, y_test_win)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        print('train_loader batch: ', len(train_loader), 'test_loader batch: ', len(test_loader))
        return train_loader, val_loader, test_loader


