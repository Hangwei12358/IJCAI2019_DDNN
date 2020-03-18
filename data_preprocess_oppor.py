# encoding=utf-8
"""
    Created on 10:38 2018/12/17
    @author: Hangwei Qian
    Adapted from: https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs

"""
import numpy as np
import torch
import pickle as cp
from pandas import Series
from torch.utils.data import Dataset, DataLoader
from utils import get_sample_weights, opp_sliding_window

NUM_FEATURES = 77

class data_loader_oppor(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)

def load_data_oppor():
    import os
    data_dir = './data/'
    saved_filename = 'oppor_processed.data'
    if os.path.isfile( data_dir + saved_filename ) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X_train = data[0][0]
        y_train = data[0][1]
        X_validation = data[1][0]
        y_validation = data[1][1]
        X_test = data[2][0]
        y_test = data[2][1]
    else:
        str_folder = data_dir + 'OpportunityUCIDataset/dataset/'
        FILES_TRAIN = [
            'S1-Drill.dat',
            'S1-ADL1.dat',
            'S1-ADL3.dat',
            'S1-ADL4.dat',
            'S1-ADL5.dat',

            'S2-Drill.dat',
            'S2-ADL1.dat',
            'S2-ADL2.dat',
            'S2-ADL5.dat',

            'S3-Drill.dat',
            'S3-ADL1.dat',
            'S3-ADL2.dat',
            'S3-ADL5.dat',

            'S4-Drill.dat',
            'S4-ADL1.dat',
            'S4-ADL2.dat',
            'S4-ADL3.dat',
            'S4-ADL4.dat',
            'S4-ADL5.dat'
        ]
        FILES_VALIDATION = [
            'S1-ADL2.dat'
        ]
        FILES_TEST = [
            'S2-ADL3.dat',
            'S2-ADL4.dat',
            'S3-ADL3.dat',
            'S3-ADL4.dat'
        ]

        label = "gestures"
        print('\nProcessing train dataset files...\n')
        FILES_TRAIN = [str_folder + a for a in FILES_TRAIN]
        FILES_VALIDATION = [str_folder + a for a in FILES_VALIDATION]
        FILES_TEST = [str_folder + a for a in FILES_TEST]
        X_train, y_train = load_data_files(label, FILES_TRAIN)
        print('\nProcessing validation dataset files...\n')
        X_validation, y_validation = load_data_files(label, FILES_VALIDATION)
        print('\nProcessing test dataset files...\n')
        X_test, y_test = load_data_files(label, FILES_TEST)
        print("Final datasets with size: | train {0}| validation {1} | test {2} | ".format(X_train.shape, X_validation.shape, X_test.shape))

        obj = [(X_train, y_train), (X_validation, y_validation), (X_test, y_test)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def load_data_files(label, data_files):
    """Loads specified data files' features (x) and labels (y)

    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    :param data_files: list of strings
        Data files to load.
    :return: numpy integer matrix, numy integer array
        Loaded sensor data, segmented into features (x) and labels (y)
    """

    data_x = np.empty((0, NUM_FEATURES))
    data_y = np.empty((0))

    for filename in data_files:
        try:
            # data = np.loadtxt(BytesIO(zipped_dataset.read(filename)))
            data = np.loadtxt(filename)
            print('... file {0}'.format(filename))
            x, y = process_dataset_file(data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ERROR: Did not find {0} in zip file'.format(filename))

    return data_x, data_y


def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data = select_columns_opp(data)

    # Colums are segmentd into features and labels
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation (a.k.a. filling in NaN)
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    data_x = normalize(data_x)

    return data_x, data_y



def select_columns_opp(data):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: tuple((numpy integer 2D matrix, numpy integer 1D matrix))
        (Selection of features (N, f), feature_is_accelerometer (f,) one-hot)
    """

    # In term of column_names.txt's ranges: excluded-included (here 0-indexed)
    features_delete = np.arange(0, 37)
    # features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(46, 50)])
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])

    # Deleting some signals to keep only the 113 of the challenge
    data = np.delete(data, features_delete, 1)

    return data


def divide_x_y(data, label):
    """Segments each sample into (time+features) and (label)

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, 0:NUM_FEATURES]
    # Choose labels type for y, the last two cols are both labels
    if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, NUM_FEATURES]  # Locomotion label
    elif label == 'gestures':
        data_y = data[:, (NUM_FEATURES+1)]  # Gestures label

    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted, no need to change class 0
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y



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



def load_fixed_slidwin_balancedUp(dataset="oppor", batch_size=64, SLIDING_WINDOW_LEN=30, SLIDING_WINDOW_STEP=15):
    if dataset=="oppor":

        x_train, y_train, x_validation, y_validation, x_test, y_test = load_data_oppor()  # (557963, 113), (557963,), (118750, 113), (118750, )

        x_train_win, y_train_win = opp_sliding_window(x_train, y_train, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        x_validation_win, y_validation_win = opp_sliding_window(x_validation, y_validation, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        x_test_win, y_test_win = opp_sliding_window(x_test, y_test, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

        unique_ytrain, counts_ytrain = np.unique(y_train_win, return_counts=True)

        weights = 100.0 / torch.Tensor(counts_ytrain)
        weights = weights.double()
        sample_weights = get_sample_weights(y_train_win, weights)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_set = data_loader_oppor(x_train_win, y_train_win)
        validation_set = data_loader_oppor(x_validation_win, y_validation_win)
        test_set = data_loader_oppor(x_test_win, y_test_win)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        print('train_loader batch: ', len(train_loader), 'validation_loader batch: ', len(validation_loader), 'test_loader batch: ', len(test_loader))
        return train_loader, validation_loader, test_loader
