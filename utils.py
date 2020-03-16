# encoding=utf-8
"""
    Created on 2018/12/10
    @author: Hangwei Qian
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sliding_window import sliding_window
import torch

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


def opp_sliding_window(data_x, data_y, ws, ss): # window size, step size
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def get_sample_weights(y, weights):
    '''
    to assign weights to each sample
    '''
    label_unique = np.unique(y)
    sample_weights = []
    for val in y:
        idx = np.where(label_unique == val)
        sample_weights.append(weights[idx])
    return sample_weights


def get_scale_matrix(M, N):
    s1 = torch.ones((N, 1)) * 1.0 / N
    s2 = torch.ones((M, 1)) * -1.0 / M
    return torch.cat((s1, s2), 0)

def mmd_custorm(sample, decoded, sigma=[1]):
    # decoded = Variable(decoded).to(device)
    X = torch.cat((decoded, sample), 0)
    XX = torch.matmul(X, X.t())
    X2 = torch.sum(X * X, 1, keepdim=True)
    exp = XX - 0.5 * X2 - 0.5 * X2.t()

    M = decoded.size()[0]
    N = sample.size()[0]
    s = get_scale_matrix(M, N)
    S = torch.matmul(s, s.t())

    loss = 0
    for v in sigma:
        kernel_val = torch.exp(exp / v)
        kernel_val = kernel_val.cpu()
        loss += torch.sum(S * kernel_val)

    loss_mmd = torch.sqrt(loss)
    return loss_mmd

def measure_event_frame(predicted_label_segment, lengths_varying_segment, true_label_segment):
    """
    this function returns the correct measurements (both frame- and event-level) for chunk-based prediction on activity
    notice that 'macro' option in sklearn does not return the desired weighted maF; therefore 'weighted' option is used instead
    """
    event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    n_event = true_label_segment.size(0)
    event_acc = ((predicted_label_segment == true_label_segment).sum()).double() * 100 / n_event
    event_miF = f1_score(true_label_segment, predicted_label_segment, average='micro') * 100
    event_maF = f1_score(true_label_segment, predicted_label_segment, average='weighted') * 100

    # create frame-based vectors
    n_frame = sum(lengths_varying_segment)
    predicted_label_frame, true_label_frame = torch.LongTensor(), torch.LongTensor()
    predicted_label_frame = torch.cat([torch.cat((predicted_label_frame, predicted_label_segment[i].repeat(lengths_varying_segment[i],1)), dim=0) for i in range(n_event)])
    assert predicted_label_frame.shape[0] == n_frame

    true_label_frame = torch.cat([torch.cat((true_label_frame, true_label_segment[i].repeat(lengths_varying_segment[i],1)), dim=0) for i in range(n_event)])
    assert true_label_frame.shape[0] == n_frame

    frame_acc = ((predicted_label_frame == true_label_frame).sum()).double() * 100 / n_frame.double()
    frame_miF = f1_score(true_label_frame, predicted_label_frame, average='micro') * 100
    frame_maF = f1_score(true_label_frame, predicted_label_frame, average='weighted') * 100

    event_acc = event_acc.item()
    frame_acc = frame_acc.item()
    return event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF