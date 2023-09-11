import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from wptt_line import get_dataset


# txt文件padding
def Max_feature_padding(read_file):
    count = len(open('D:\\pyproject\\TCRNdemo\\'+read_file, 'r').readlines())
    if count < 2998:
        for i in range(2998 - count):
            with open('D:\\pyproject\\TCRNdemo\\'+read_file, 'a+') as fpp:
                fpp.write('0' + ',' + '0' + '\n')


def distance(p0, q0, p1, q1):
    d = math.sqrt((p1 - p0) * (p1 - p0) + (q1 - q0) * (q1 - q0))
    return d


def redundant_point_d(p0, q0, p1, q1, d):
    if distance(p0, q0, p1, q1) > d:
        return True
    else:
        return False


def trajectory_preprocessing(x, y, d):
    length = len(x)
    print('原始点的个数:' + str(length))
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x1.append(x[0])
    y1.append(y[0])
    # 删除所有冗余点
    for i in range(1, length):
        if redundant_point_d(x[i], y[i], x[i - 1], y[i - 1], d) and (x[i] != x1[-1] and y[i] != y1[-1]) and len(
                x1) <= 2:
            x1.append(x[i])
            y1.append(y[i])
        elif redundant_point_d(x[i], y[i], x[i - 1], y[i - 1], d) and (x[i] != x1[-1] and y[i] != y1[-1]) and (
                x[i] != x1[-2] and y[i] != y1[-2]):
            x1.append(x[i])
            y1.append(y[i])
        else:
            continue
    print('删除后点的个数:' + str(len(x1)))
    # 坐标归一化
    '''x2.append(x1[0])
    y2.append(y1[0])'''
    return x1, y1


# 0-1归一化
def normLization01(a, b):
    x_max = np.max(a)
    x_min = np.min(a)
    y_max = np.max(b)
    y_min = np.min(b)
    c = (x_max - x_min) / (y_max - y_min)
    x = []
    y = []
    for i in range(len(a)):
        x.append(c * ((a[i] - x_min) / (x_max - x_min)))
        y.append((b[i] - y_min) / (y_max - y_min))
    return x, y


# -1-1归一化
def normLization_11(a, b):
    x_max = np.max(a)
    x_min = np.min(a)
    y_max = np.max(b)
    y_min = np.min(b)
    c = (x_max - x_min) / (y_max - y_min)
    x = []
    y = []
    for i in range(len(a)):
        x.append(c * (2 * (a[i] - x_min) / (x_max - x_min) - 1))
        y.append(2 * (b[i] - y_min) / (y_max - y_min) - 1)
    return x, y


# 删除冗余点和等距离采样
def Removeredundantpointswithequidistantsampling(x, y):
    x, y = trajectory_preprocessing(x, y, 0.05)
    i = 1
    dis_min = 1000000
    while i < len(x):
        if np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) < dis_min:
            dis_min = np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        i += 1
    i = 1
    u = []
    v = []
    while i < len(x):
        if x[i - 1] <= x[i]:
            X = [x[i - 1], x[i]]
            Y = [y[i - 1], y[i]]
        else:
            X = [x[i], x[i - 1]]
            Y = [y[i], y[i - 1]]
        x_val = np.linspace(X[0], X[1], int(np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) / dis_min))
        yinterp = np.interp(x_val, X, Y)
        if x[i - 1] <= x[i]:
            for j in range(len(x_val)):
                u.append(x_val[j])
                v.append(yinterp[j])
        else:
            j = len(x_val) - 1
            while j >= 0:
                u.append(x_val[j])
                v.append(yinterp[j])
                j -= 1
        i += 1
    return u, v


# 弯曲矫正
def Bendingcorrection(u, v):
    avex = []
    avey = []
    s = 0
    sux = 0
    suy = 0
    for i in range(len(u)):
        s += 1
        if s == 4:
            avex.append(sux / 4)
            avey.append(suy / 4)
            s = 0
            sux = 0
            suy = 0
        sux += u[i]
        suy += v[i]

    ax = np.array(avex)
    ay = np.array(avey)
    z1 = np.polyfit(ax, ay, 2)
    p1 = np.poly1d(z1)
    X2 = []
    Y2 = []
    avex2 = []
    avey2 = []
    s2 = 0
    sux2 = 0
    suy2 = 0
    for i in range(len(u)):
        Y2.append(v[i] - p1(u[i]))
        X2.append(u[i])
    # for i in range(len(X2)):
    #     s2 += 1
    #     if s2 == 4:
    #         avex2.append(sux2/4)
    #         avey2.append(suy2/4)
    #         s2 = 0
    #         sux2 = 0
    #         suy2 = 0
    #     sux2 += X2[i]
    #     suy2 += Y2[i]
    # ax2 = np.array(avex2)
    # ay2 = np.array(avey2)
    # z2 = np.polyfit(ax2, ay2, 2)
    # p2 = np.poly1d(z2)
    # fx = p1(ax)
    # fx2 = p2(ax2)
    # plt.plot(X2, Y2, color='black')
    # plt.plot(u, v, color='black')
    # plt.plot(ax2, fx2, color='green')
    # plt.plot(ax, fx, color='green')
    # plt.ylim(-100, 30)
    # plt.xlim(0, 80)
    # plt.show()
    return X2, Y2


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    f = open('../data/txt/old_casia_2247/Train_xy_txt/001-P16_02.txt')
    a = []
    b = []
    f.readline()
    for l in f.readlines():
        a.append(float(l.split(',')[0].strip('\n')))
        b.append(float(l.split(',')[1].strip('\n')))
    start = time.time()
    x, y = normLization_11(a, b)
    u, v = Removeredundantpointswithequidistantsampling(x, y)
    X2, Y2 = Bendingcorrection(u, v)
    end = time.time()
    print(end - start)
    plt.figure(figsize=(20, 2))
    plt.plot(x, y)
    plt.show()
    plt.figure(figsize=(20, 2))
    plt.plot(X2, Y2)
    # plt.plot(X2, Y2)
    plt.show()






