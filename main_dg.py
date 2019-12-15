# encoding=utf-8
"""
    Created on 2018/12/10
    @author: Hangwei Qian
"""
import matplotlib
matplotlib.use('Agg')

import network_dg as net
import data_preprocess_dg
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import argparse
from sklearn.metrics import f1_score
import torch.nn.functional as F
from constants import *
from custom_funcs import *
import os
if not os.path.exists(model_path):
    os.mkdir(model_path)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = []
acc_all = []

def get_scale_matrix(M, N):
    s1 = torch.ones((N, 1)) * 1.0 / N
    s2 = torch.ones((M, 1)) * -1.0 / M
    return torch.cat((s1, s2), 0)

def mmd_custorm(sample, decoded, sigma=[1]):
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

def train_dg_fixed(model, optimizer, train_loader, test_loader, now_model_name, args):
    feature_dim = args.n_feature  # for dg dataset
    n_batch = len(train_loader.dataset) // args.batch_size
    criterion = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    for e in range(args.n_epoch):
        if e>0 and e%50 == 0:
            plot(result_name)

        model.train()
        correct, total_loss = 0, 0
        total = 0

        for index, (sample, target) in enumerate(train_loader):
            # get the info of the sample to check sampler's function on the 1st iteration
            if index == 0:
                unique_ytrain, counts_ytrain = np.unique(target, return_counts=True)
                print(index, 'th ', 'sample label distribution: ', dict(zip(unique_ytrain, counts_ytrain)))

            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            now_len = sample.shape[1]
            sample = sample.view(-1, feature_dim, now_len)
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            output, out_decoder = model(sample)

            loss_classify = criterion(output, target)
            loss_ae = criterion_ae(sample.view(sample.size(0), -1), out_decoder)
            loss_mmd = mmd_custorm(sample.view(sample.size(0), -1), out_decoder, [args.sigma])
            loss_mmd = loss_mmd.to(DEVICE).float()
            loss = loss_classify + LOSS_FN_WEIGHT * loss_ae + args.weight_mmd*loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

            if index % 20 == 0:
                tqdm.tqdm.write('Epoch: [{}/{}], Batch: [{}/{}], loss_ae:{:.4f}, loss_mmd:{:.4f}, loss_classify:{:.4f}, loss_total:{:.4f}'.format(e + 1, args.n_epoch, index + 1, n_batch,
                                                                                   loss_ae.item(), loss_mmd.item(), loss_classify.item(), loss.item()))
        acc_train = float(correct) * 100.0 / (args.batch_size * n_batch)
        tqdm.tqdm.write(
            'Epoch: [{}/{}], loss: {:.4f}, train acc: {:.2f}%'.format(e + 1, args.n_epoch, total_loss * 1.0 / n_batch, acc_train))

        # Testing
        model.train(False)
        with torch.no_grad():
            correct, total = 0, 0
            event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            predicted_label_segment, lengths_varying_segment, true_label_segment = torch.LongTensor(), torch.LongTensor(), torch.LongTensor()
            for sample, target in test_loader:
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                now_len = sample.shape[1]
                # this line would cause error since the batch of last iteration does not have batch_size entries. so use DropLast = True when prep for dataloader
                sample = sample.view(-1, feature_dim, now_len)
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                output, out_decoder = model(sample)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
                lengths_varying = [sample.shape[2]] * sample.shape[0]
                lengths_varying = torch.LongTensor(lengths_varying)
                predicted_label_segment = torch.LongTensor(torch.cat((predicted_label_segment, predicted.cpu()), dim=0))
                lengths_varying_segment = torch.LongTensor(torch.cat((lengths_varying_segment, lengths_varying), dim=0))
                true_label_segment = torch.LongTensor(torch.cat((true_label_segment, target.cpu()), dim=0))

        # calculate different measurements
        # event: accuracy, micro-F1, macro-F1
        # frame: accuracy, micro-F1, macro-F1
        event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = measure_event_frame(predicted_label_segment, lengths_varying_segment, true_label_segment)
        # best accuracy record
        acc_all.append([event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF])
        acc_all_T = np.array(acc_all).T.tolist()

        best_e_miF = max([row[1] for row in acc_all])
        best_iter = acc_all_T[1].index(best_e_miF) + 1

        best_e_acc = acc_all[best_iter-1][0]
        best_e_maF = acc_all[best_iter-1][2]
        best_f_acc = acc_all[best_iter-1][3]
        best_f_miF = acc_all[best_iter-1][4]
        best_f_maF = acc_all[best_iter-1][5]
        if sum(predicted_label_segment) == 0:
            print('Note: All predicted labels are 0 in this epoch!\n')

        tqdm.tqdm.write(
            'Epoch: [{}/{}], e acc:{:.2f}%, e_miF:{:.2f}%, e maF:{:.2f}%, f acc:{:.2f}%, f miF:{:.2f}%, f maF:{:.2f}%, best acc:{:.2f}%, iter:{}'.format(
                e + 1, args.n_epoch, event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF, best_e_acc,
                best_iter))
        result.append([acc_train, event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF, best_e_acc, best_iter])
        result_np = np.array(result, dtype=float)
        np.savetxt(result_name, result_np, fmt='%.2f', delimiter=',')

    # to log best performance to file without overwriting
    return best_e_acc, best_e_miF, best_e_maF, best_f_acc, best_f_miF, best_f_maF, best_iter


# define the arguments, added by hangwei
parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--now_model_name', type=str, default='ae_spatial_LSTM_CNN',
                    help='the type of model, default autoencoder+LSTM')
parser.add_argument('--n_lstm_layer', type=int, default=2, help='number of lstm layers,default 2')
parser.add_argument('--n_lstm_hidden', type=int, default=64, help= 'number of lstm hidden dim, default 64')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=500, help='number of training epochs')
parser.add_argument('--slidwin', type=str, default='balancedUp', choices=['fixed', 'varying', 'balancedUp', 'balancedDown', 'numberEqual'], help='fixed or varying sliding window length for segmentation')
parser.add_argument('--dataset', type=str, default='dg', choices=['oppor', 'ucihar', 'pamap2', 'dg'], help='name of dataset')

# arguments to make code more concise across different datasets
parser.add_argument('--n_feature', type=int, default=9, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=32, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=2, help='number of class')
parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
parser.add_argument('--sigma', type=float, default=1, help='parameter of mmd')
parser.add_argument('--weight_mmd', type=float, default=1.0, help='weight of mmd loss')

if __name__ == '__main__':
    torch.manual_seed(10)
    args = parser.parse_args()
    if args.slidwin == "balancedUp":
        print('FIXED sliding window length mode\n')
        train_loader, val_loader, test_loader = data_preprocess_dg.load_dataset_dg(batch_size=args.batch_size, SLIDING_WINDOW_LEN=32, SLIDING_WINDOW_STEP=16)

        if args.now_model_name == "ae_spatial_LSTM_CNN":
            model = net.ae_spatial_LSTM_CNN(args).to(DEVICE)
        else:
            print('model not available!\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result_name = 'results/' + args.dataset + '/'+ args.slidwin + '_'+str(args.n_epoch) +'_' + str(args.batch_size) +'_' + args.now_model_name + '_'+str(args.n_lstm_hidden)+'_' + str(args.n_lstm_layer)+'.csv'

        if not os.path.exists('results/' + args.dataset):
            os.makedirs('results/' + args.dataset)
        # create an empty csv file if not exist
        if not os.path.isfile(result_name):
            with open(result_name, 'w') as my_empty_csv:
                pass

        best_e_acc, best_e_miF, best_e_maF, best_f_acc, best_f_miF, best_f_maF, best_iter = train_dg_fixed(model, optimizer, train_loader, test_loader, result_name, args)
        # log the best results and corresponding configs in a single file for each dataset
        dataset_name = "{}".format(args.dataset)
        # TODO: the output of numbers contain many digits after ., which can be reduced to 3 digits
        with open('results/{}/best_result_{}.txt'.format(dataset_name, dataset_name), 'a') as f:
            f.write('now_model_name: '+args.now_model_name + '\t e_acc: '+str(best_e_acc) + '\t e_miF: '+str(best_e_miF) + '\t e_maF: '+str(best_e_maF)
                    + '\t f_acc: '+str(best_f_acc)  + '\t f_miF: '+str(best_f_miF)  + '\t f_maF: '+str(best_f_maF) + '\t best_iter: '+str(best_iter)
                    +'\t n_lstm_hidden: ' + str(args.n_lstm_hidden) + '\t n_lstm_layer: ' + str(args.n_lstm_layer) +
                    '\t batch_size: ' + str(args.batch_size) + '\t n_epoch: ' + str(args.n_epoch) +' '+ args.slidwin
                    + '\t d_AE: ' + str(args.d_AE) + '\t weight_mmd: ' + str(args.weight_mmd) + '\t sigma: ' + str(args.sigma) + '\n\n')
        plot(result_name)
    else:
        print('Unknown sliding window mode! \n')

