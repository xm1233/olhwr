import time

import cv2.cv2 as cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import random

import win32file

from chinese_char import chars_list2
from scipy.spatial.distance import pdist

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''fp = open('D:\\CASIA-OLHWDB2\\Train_xy_txt\\795-P16_6.txt')
x = []
y = []
for line in fp.readlines():
    x.append(float(line.split(',')[0]))
    y.append(float(line.split(',')[1]))
plt.plot(x,y)
plt.ylim(-100,0)
plt.xlim(0,80)
plt.show()'''


def REplace(txt_file):
    fp = open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+txt_file, 'r')
    with open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+txt_file, 'a+') as f:
        for line in fp.readlines():
            f.write(line.strip('\n').replace('\x00', ' ').replace(' ', '') + '\n')


def feature_create(read_file, output_file):
    for filename in os.listdir('D:\\pyproject\\TCRNdemo\\data\\txt\\'+read_file+'\\'):
        with open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+read_file+'\\' + filename, "r") as f:
            X = []
            Y = []
            for line in f.readlines():
                X.append(float(line.split(',')[0]))
                Y.append(float(line.split(',')[1]))
            with open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+output_file+'\\' + filename.split('.')[0] + 'c' + '.txt',
                      'a+') as fp:
                for i in range(len(X)):
                    if i == 0 or i == 1:
                        fp.write('0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + '\n')
                    elif i == len(X) - 1 or i == len(X) - 2:
                        fp.write(str(X[i] - X[i - 1]) + ' ' + str(
                            Y[i] - Y[i - 1]) + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + '\n')
                    else:
                        fp.write(str(X[i] - X[i - 1]) + ' ' + str(Y[i] - Y[i - 1]) + ' ' +
                                 str((Y[i + 1] - Y[i - 1]) / np.sqrt(
                                     (X[i - 1] - X[i + 1]) ** 2 + (Y[i - 1] - Y[i + 1]) ** 2)) + ' ' + str(
                            (X[i + 1] - X[i - 1]) / np.sqrt(
                                (X[i - 1] - X[i + 1]) ** 2 + (Y[i - 1] - Y[i + 1]) ** 2)) + ' ')
                        x = (X[i] - X[i - 2], Y[i] - Y[i - 2])
                        y = (X[i + 2] - X[i], Y[i + 2] - Y[i])
                        cos = 1 - pdist([x, y], 'cosine')
                        sin = np.sqrt(1 - cos ** 2)
                        fp.write(str(cos.item()) + ' ' + str(sin.item()) + '\n')


def x_label_data():
    a = []
    b = []
    for filename in os.listdir('D:\\pyproject\\TCRNdemo\\data\\txt\\Train_feature_txt\\'):
        a.append(filename)
    with open('D:\\pyproject\\TCRNdemo\\data\\txt\\Train_labels_all.txt', 'r') as f:
        for line in f.readlines():
            b.append(str(line).strip('\n'))
    with open('D:\\pyproject\\TCRNdemo\\data\\txt\\Train_all.txt', 'a+') as f:
        for i in range(len(a)):
            f.write(a[i] + "$" + b[i] + "\n")


def Max_feature_padding(read_file, output_file):
    max_length = 0
    for f in os.listdir('D:\\pyproject\\TCRNdemo\\data\\txt\\'+read_file+'\\'):
        count = len(open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+read_file+'\\' + f, 'r').readlines())
        if count > max_length:
            s = f
            max_length = count
    print(max_length, s)

    for f in os.listdir('D:\\pyproject\\TCRNdemo\\data\\txt\\'+output_file+'\\'):
        count = len(open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+output_file+'\\' + f, 'r').readlines())
        if count < max_length:
            for i in range(max_length - count):
                with open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+output_file+'\\' + f, 'a+') as fpp:
                    fpp.write('0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + '\n')


def Replace_nan(file_name):
    for filename in os.listdir('D:\\pyproject\\TCRNdemo\\data\\txt\\'+file_name+'\\'):
        file_data = ""
        with open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+file_name+'\\' + filename, 'r') as f:
            for line in f:
                if 'nan' in line:
                    line = line.replace('nan', '0')
                file_data += line
        with open('D:\\pyproject\\TCRNdemo\\data\\txt\\'+file_name+'\\' + filename, 'w') as f:
            f.write(file_data)


def load_traj_from_file(filename):
    f = open(filename, 'r')

    print(f.readline())
    trajs = np.loadtxt(f, dtype='float32')

    f.close()
    return trajs


def plot_points_3d(traj, info='b', simple=True):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if simple:
        def format_func(value, tick_number):
            return ''

        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.grid(True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.zaxis.set_major_formatter(plt.FuncFormatter(format_func))

        # plt.axis('off')
        ax.w_xaxis.line.set_color("skyblue")
        ax.w_yaxis.line.set_color("skyblue")
        ax.w_zaxis.line.set_color("skyblue")
        # ax.w_yaxis.set_pane_color((0.8, .8, .8))
        # ax.w_zaxis.set_pane_color((0.7, .7, .7))
        # ax.w_xaxis.set_pane_color((0.9, .9, .9))

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], info)
    plt.show()


def plot_points(traj, info='k-', off_axie=True):
    out = np.max(traj, axis=0) - np.min(traj, axis=0)
    if out.shape[0] == 3:
        width, height, _ = out
    else:
        width, height = out
    nw = width / height * 2
    plt.figure(figsize=(nw, 1))
    plt.plot(traj[:, 0], traj[:, 1], info, linewidth=1.0, marker='.', markersize=3)
    if off_axie:
        plt.axis('off')
        # plt.savefig('../o.png', dpi=1000, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    file = open('../data/txt/extra_data/iahw_extra_data_xy/iahw_90.txt')
    x = []
    y = []
    for l in file.readlines():
        x.append(float(l.split(',')[0].strip('\n')))
        y.append(float(l.split(',')[1].strip('\n')))
    plt.figure(figsize=(20, 2))
    plt.axis('off')
    plt.plot(x, y, 'k-', linewidth=1)

    # plt.savefig('../ia.png', dpi=600, bbox_inches='tight', pad_inches=-0.1)
    plt.show()



