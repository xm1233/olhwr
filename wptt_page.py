import matplotlib.pyplot as plt
import numpy as np
import struct
import os
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def coordinate_norm(x, y):
    m = 0.0
    np = 0.0
    nq = 0.0
    deltat = 0.0
    length = len(x)
    for i in range(length - 1):
        d = distance(x[i], y[i], x[i + 1], y[i + 1])
        np += d * (x[i] + x[i + 1])
        nq += d * (y[i] + y[i + 1])
        m += d

    mup = float(0.5 * np / m)
    muq = float(0.5 * nq / m)

    for i in range(length - 1):
        d = distance(x[i], y[i], x[i + 1], y[i + 1])

        deltat += d * (
                (x[i + 1] - mup) * (x[i + 1] - mup) + (x[i + 1] - mup) * (x[i] - mup) + (x[i] - mup) * (x[i] - mup))

    deltap = math.sqrt(float(1 / 3) * deltat / m)
    for i in range(len(x)):
        x[i] = (x[i] - mup) / deltap
        y[i] = (y[i] - muq) / deltap
    return x, y


if __name__ == '__main__':
    root = 'D:\\WPTT\\Competition13wptt\\'
    filename = 'C008-p18.wptt'

    i = 1
    s = []

    with open(root+filename, 'rb') as f:
        headsize = np.fromfile(f, dtype='uint8', count=4)
        headsize = headsize[0] + (headsize[1] << 8)+(headsize[2] << 16)+(headsize[3] << 24)
        print(headsize)

        Format_code = np.fromfile(f, dtype='uint8', count=8)
        format_code = ''
        for i in range(4):
            format_code += chr(Format_code[i])
        print(format_code)

        Illustration_list = np.fromfile(f, dtype='uint8', count=headsize-54)
        illustration = ''
        for i in Illustration_list:
            illustration += struct.pack('I', i).decode('gbk', 'ignore')[0]
        print("illustration:", illustration)

        Code_type = np.fromfile(f, dtype='uint8', count=20)
        Code_length = np.fromfile(f, dtype='uint8', count=2)
        print("Code_length:", Code_length)
        Code_length = sum([j << (i*8) for i, j in enumerate(Code_length)])
        Datatype_lisy = np.fromfile(f, dtype='uint8', count=20)

        sample_length = np.fromfile(f, dtype='uint8', count=4)
        print("sample_length:", sum([j << (i*8) for i, j in enumerate(sample_length)]))
        page_index = np.fromfile(f, dtype='uint8', count=4)
        print("page_index:", sum([j << (i*8) for i, j in enumerate(page_index)]))

        stroke_num = np.fromfile(f, dtype='uint8', count=4)
        stroke_num = sum([j << (i*8) for i, j in enumerate(stroke_num)])
        print("stroke_num:", stroke_num)

        for stroke_idx in range(stroke_num):
            Points_num = np.fromfile(f, dtype='uint8', count=2)
            Points_num = sum([j << (i*8) for i, j in enumerate(Points_num)])
            X = []
            Y = []
            for point_idx in range(Points_num):
                Coordinates_x = np.fromfile(f, dtype='uint8', count=2)
                Coordinates_x = sum([j << (i*8) for i, j in enumerate(Coordinates_x)])
                Coordinates_y = np.fromfile(f, dtype='uint8', count=2)
                Coordinates_y = sum([j << (i*8) for i, j in enumerate(Coordinates_y)])
                X.append(0.01*Coordinates_x)
                Y.append(-0.01*Coordinates_y)
            # X1, Y1 = trajectory_preprocessing(X, Y, 0.12)
            plt.plot(X, Y, 'k-', linewidth=0.3)
        plt.axis('off')
        # plt.savefig('olhw.jpg', dpi=600, bbox_inches='tight', pad_inches=-0.1)
        plt.show()

        line_num = np.fromfile(f, dtype='uint8', count=2)
        line_num = sum([j << (i*8) for i, j in enumerate(line_num)])

        print("line_num:", line_num)

        for line_idx in range(line_num):

            line_stroke_num = np.fromfile(f, dtype='uint8', count=2)
            line_stroke_num = sum([j << (i*8) for i, j in enumerate(line_stroke_num)])
            print("The {} line_stroke_num: {}".format(line_idx+1, line_stroke_num))
            for j in range(line_stroke_num):
                line_stroke_idx = np.fromfile(f, dtype='uint8', count=2)
                line_stroke_idx = sum([j << (i*8) for i, j in enumerate(line_stroke_idx)])

            line_char_num = np.fromfile(f, dtype='uint8', count=2)
            line_char_num = sum([j << (i*8) for i, j in enumerate(line_char_num)])
            print("The {} line_char_num: {}".format(line_idx+1, line_char_num))
            line_label = ''
            for line_char_idx in range(line_char_num):
                tag_code = np.fromfile(f, dtype='uint8', count=Code_length)
                tag_code = sum([j << (i*8) for i, j in enumerate(tag_code)])
                label = struct.pack('I', tag_code).decode('gbk', 'ignore')[0]

                line_label += label
            print(line_label)
