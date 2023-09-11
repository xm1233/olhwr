import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.chinese_char import chars_list_casia, chars_list2
from utils.prepro import normLization01, normLization_11
from PyQt5.QtCore import pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QPainter, QPainterPath
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def show(file):
    f = open('./data/txt/' + file, 'r')
    X = []
    Y = []
    for l in f.readlines():
        X.append(float(l.split(',')[0]))
        Y.append(float(l.split(',')[1].strip('\n')))
    d_x = np.max(X) - np.min(Y)
    d_y = np.max(Y) - np.min(Y)
    nw = d_x / d_y * 2
    plt.figure(figsize=(20, 3))
    # plt.plot(X, Y, linewidth=0.8, marker='.', markersize=2)
    plt.plot(X, Y, linewidth=0.8)
    plt.axis('off')
    plt.show()


def iahw_out_dict():
    file = open('./data/txt/iahcc_label_dict.txt')
    num = []
    char = []
    dt = {}
    for l in file.readlines():
        t = l.split('\t')[0]
        if ',' in t:
            t = t.split(',')[0] + t.split(',')[1]
        num.append(int(t))
        char.append(l.split('\t')[1])
    for i in range(len(num)):
        dt.setdefault(num[i], char[i])
    return dt


def hwdb_out_dict():
    file = open('./data/txt/hwdb_char_list.txt', encoding='utf-8')
    num = []
    t = 1
    char = []
    dt = {}
    for l in file.readlines():
        num.append(t)
        t += 1
        char.append(l.split('\n')[0])
    for i in range(len(num)):
        dt.setdefault(num[i], char[i])
    return dt


if __name__ == '__main__':

    # f = open('./data/txt/old_casia_2247/train_data.txt', 'r', encoding='utf-8')
    # fe = open('./data/txt/old_casia_2247/cmp13_data.txt', 'r', encoding='utf-8')
    # x = []
    # y = []
    # for l in f.readlines():
    #     for c in l.split('$')[1].strip('\n'):
    #         x.append(c)
    # for l in fe.readlines():
    #     for c in l.split('$')[1].strip('\n'):
    #         y.append(c)
    # cnt = 0
    # for i in range(len(y)):
    #     if y[i] not in x:
    #         cnt += 1
    # print(len(y), cnt)

    # data = torch.load('./data/txt/iahcc/train_data_5.pt')
    # iahw_dt = iahw_out_dict()
    # hwdb_dt = hwdb_out_dict()
    # c = 0
    # while c < 1000:
    #     cnt = 0
    #     x = []
    #     y = []
    #     total = random.randint(14, 18)
    #     x_max = 0
    #     s = ""
    #     while cnt < total:
    #         k = random.randint(1, len(data))
    #         d = data[k]['data']
    #         label = data[k]['label']
    #         if iahw_dt[label] in chars_list_casia:
    #             cnt += 1
    #             s += iahw_dt[label]
    #             l = len(d[0])
    #             distance = 4 * random.random()
    #             for i in range(l):
    #                 x.append(d[0][i] + x_max + distance)
    #                 y.append(d[1][i])
    #             x_max = max(x) + distance
    #     print(s)
    #     x, y = normLization_11(x, y)
    #     if len(x) <= 2998:
    #         with open('./data/txt/extra_data/iahw_extra_data_xy/iahw_' + str(c+3000) + '.txt', 'a+') as fout:
    #             for i in range(len(x)):
    #                 fout.write(str(x[i]) + ',' + str(y[i]) + '\n')
    #         txt_file = open('./data/txt/iahw_extra_data.txt', 'a+')
    #         txt_file.write('iahw_'+str(c+3000)+'.txt$'+s+'\n')
    #         c += 1

    # x, y = normLization01(x, y)
    # plt.plot(x, y)
    # plt.show()

    for fname in os.listdir('./data/txt/extra_data/iahw_extra_data_xy/'):
        x = []
        y = []
        file = open('./data/txt/extra_data/iahw_extra_data_xy/'+fname, 'r')
        for l in file.readlines():
            x.append(float(l.split(',')[0]))
            y.append(float(l.split(',')[1].strip()))
        count = len(open('./data/txt/extra_data/iahw_extra_data_xy/'+fname, 'r').readlines())
        if count < 2998:
            for i in range(2998 - count):
                x.append(0)
                y.append(0)
        for i in range(len(x)):
            with open('./data/txt/extra_data/iahw_extra_feature/'+fname,'a+') as fp:
                fp.write(str(x[i])+','+str(y[i])+'\n')



