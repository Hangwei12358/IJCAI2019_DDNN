# encoding=utf-8
# this script contains some customed and stable functions to make main script concise
# by hangwei 1.18.2019

import numpy as np
import matplotlib.pyplot as plt

def plot(result_name):
    data = np.loadtxt(result_name, delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Test Accuracy', fontsize=20)

    plt.savefig(result_name[:-4]+'_figure.png')
