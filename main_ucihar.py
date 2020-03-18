# encoding=utf-8
"""
    Created on 2018/12/10
    @author: Hangwei Qian
"""
import matplotlib
matplotlib.use('Agg')

import network_ucihar as net
import data_preprocess_ucihar
import torch
import torch.nn as nn
import tqdm
import argparse
from utils import *
import os
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = []
acc_all = []
LOSS_FN_WEIGHT = 1e-5

def train_ucihar_fixed(model, optimizer, train_loader, test_loader, now_model_name, args):
    feature_dim = args.n_feature
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
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            now_len = sample.shape[3]
            sample = sample.view(-1, feature_dim, now_len)

            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

            output, out_decoder = model(sample)

            loss_classify = criterion(output, target)
            loss_ae = criterion_ae(sample.view(sample.size(0), -1), out_decoder)
            loss_mmd = mmd_custorm(sample.view(sample.size(0), -1), out_decoder)
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
                now_len = sample.shape[3]
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
    return best_e_acc, best_e_miF, best_e_maF, best_f_acc, best_f_miF, best_f_maF, best_iter


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--now_model_name', type=str, default='DDNN', help='the type of model, default DDNN')
parser.add_argument('--n_lstm_layer', type=int, default=2, help='number of lstm layers,default 2')
parser.add_argument('--n_lstm_hidden', type=int, default=128, help= 'number of lstm hidden dim, default 64')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--dataset', type=str, default='ucihar', help='name of dataset')

parser.add_argument('--n_feature', type=int, default=9, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=128, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=6, help='number of class')
parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
parser.add_argument('--sigma', type=float, default=1, help='parameter of mmd')
parser.add_argument('--weight_mmd', type=float, default=1.0, help='weight of mmd loss')


if __name__ == '__main__':
    torch.manual_seed(10)
    args = parser.parse_args()

    train_loader, test_loader = data_preprocess_ucihar.load_balancedUp(batch_size=args.batch_size)

    model = net.DDNN(args).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    result_name = 'results/' + args.dataset + '/'+str(args.n_epoch) +'_' + str(args.batch_size) +'_' + args.now_model_name + '_'+str(args.n_lstm_hidden)+'_' + str(args.n_lstm_layer)+'.csv'
    if not os.path.exists('results/' + args.dataset):
        os.makedirs('results/' + args.dataset)
    if not os.path.isfile(result_name):
        with open(result_name, 'w') as my_empty_csv:
            pass

    best_e_acc, best_e_miF, best_e_maF, best_f_acc, best_f_miF, best_f_maF, best_iter = train_ucihar_fixed(model, optimizer, train_loader, test_loader, result_name, args)

    dataset_name = "{}".format(args.dataset)
    with open('results/{}/best_result_{}.txt'.format(dataset_name, dataset_name), 'a') as f:
        f.write('now_model_name: ' + args.now_model_name + '\t e_acc: ' + str(best_e_acc) + '\t e_miF: ' + str(
            best_e_miF) + '\t e_maF: ' + str(best_e_maF)
                + '\t f_acc: ' + str(best_f_acc) + '\t f_miF: ' + str(best_f_miF) + '\t f_maF: ' + str(
            best_f_maF) + '\t best_iter: ' + str(best_iter)  + '\t d_AE: ' + str(args.d_AE)
                + '\t n_lstm_hidden: ' + str(args.n_lstm_hidden) + '\t n_lstm_layer: ' + str(args.n_lstm_layer)
                + '\t batch_size: ' + str(args.batch_size) + '\t n_epoch: ' + str(args.n_epoch) + '\n\n')
    plot(result_name)
