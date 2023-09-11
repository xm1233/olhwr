import os
import struct
import matplotlib.pyplot as plt
import numpy as np


def get_dataset(root, wptt) -> []:
    with open(root + wptt, 'rb') as f:

        szhead = np.fromfile(f, dtype='uint8', count=4)
        szhead = szhead[0] + (szhead[1] << 8) + (szhead[2] << 16) + (szhead[3] << 24)
        # 文件头大小
        # print(szhead)

        Format_code = np.fromfile(f, dtype='uint8', count=8)
        format_code = ''
        for i in range(4):
            format_code += chr(Format_code[i])
        # 模板代码
        # print(format_code)

        Illustration_list = np.fromfile(f, dtype='uint8', count=szhead - 54)
        illustration = ''
        for i in Illustration_list:
            illustration += struct.pack('I', i).decode('gbk', 'ignore')[0]
        # 说明
        # print('illustration:', illustration)

        Code_type = np.fromfile(f, dtype='uint8', count=20)
        Code_length = np.fromfile(f, dtype='uint8', count=2)
        Code_length = sum([j << (i * 8) for i, j in enumerate(Code_length)])
        # 代码长度
        # print('Code_length:', Code_length)

        Datatype_list = np.fromfile(f, dtype='uint8', count=20)
        sample_length = np.fromfile(f, dtype='uint8', count=4)
        # 例子长度
        # print('sample_length:', sum([j << (i * 8) for i, j in enumerate(sample_length)]))

        page_index = np.fromfile(f, dtype='uint8', count=4)
        page_index = sum([j << (i * 8) for i, j in enumerate(page_index)])
        # 页面索引
        # print('page_index:', page_index)

        stroke_nums = np.fromfile(f, dtype='uint8', count=4)
        stroke_nums = sum([j << (i * 8) for i, j in enumerate(stroke_nums)])
        # 笔画总数
        # print('笔画总数stroke_nums:', stroke_nums)
        strokes = []
        for stroke_idx in range(stroke_nums):
            Points_nums = np.fromfile(f, dtype='uint8', count=2)
            Points_nums = sum([j << (i * 8) for i, j in enumerate(Points_nums)])
            # 当前笔画存在多少个点
            # print('--第{}笔有多少点Points_nums:'.format(stroke_idx + 1), Points_nums)
            X = []
            Y = []
            Coor = []
            for point_idx in range(Points_nums):
                Coor_x = np.fromfile(f, dtype='uint8', count=2)
                Coor_x = sum([j << (i * 8) for i, j in enumerate(Coor_x)])
                Coor_y = np.fromfile(f, dtype='uint8', count=2)
                Coor_y = sum([j << (i * 8) for i, j in enumerate(Coor_y)])
                # 当前点的坐标
                # print('----' + str(Coor_x*0.1) + ',' + str(-0.1*Coor_y))
                Coor.append([Coor_x, Coor_y])
            strokes.append(Coor)
            # print(strokes)
        #     plt.plot(X, Y)
        # plt.show()

        line_nums = np.fromfile(f, dtype='uint8', count=2)
        line_nums = sum([j << (i * 8) for i, j in enumerate(line_nums)])
        # 总行数
        # print('总行数line_nums:', line_nums)

        lines = []
        # line_label_chars = ''
        for line_idx in range(line_nums):
            line_stroke_num = np.fromfile(f, dtype='uint8', count=2)
            line_stroke_num = sum([j << (i * 8) for i, j in enumerate(line_stroke_num)])
            # 当前行的笔画总数
            # print('第{}行笔画总数: {}'.format(line_idx + 1, line_stroke_num))

            # 当前行笔画起始位置
            line_stroke_idx = np.fromfile(f, dtype='uint8', count=2)
            line_stroke_idx = sum([j << (i * 8) for i, j in enumerate(line_stroke_idx)])

            _ = np.fromfile(f, dtype='uint8', count=2 * line_stroke_num - 2)

            line_char_num = np.fromfile(f, dtype='uint8', count=2)
            line_char_num = sum([j << (i * 8) for i, j in enumerate(line_char_num)])
            # 当前行的字符总数
            # print('第{}行字符总数: {}'.format(line_idx + 1, line_char_num))

            line_label = ''
            for line_char_idx in range(line_char_num):
                tag_code = np.fromfile(f, dtype='uint8', count=Code_length)
                tag_code = sum([j << (i * 8) for i, j in enumerate(tag_code)])
                label = struct.pack('I', tag_code).decode('gbk', 'ignore')[0]
                line_label += label

            lines.append([line_char_num, line_stroke_num, line_label,
                          strokes[line_stroke_idx:line_stroke_idx + line_stroke_num]])

        #     line_label_chars += line_label + '\n'
        # print(line_label_chars)

        return page_index, line_nums, lines


# dataset = get_dataset('dataset/OLHWDB2/Competition13wptt/', 'C001-P16')
# loader=DataLoader(dataset=dataset,batch_size=50,shuffle=True)
# print(type(dataset))
# print(type(loader))
# for i in dataset[:-1]:
#     print(i)
# for i in dataset[2]:
#     print(i)
def wpdecode():
    root = 'dataset/OLHWDB2/WPTT2.2-Train/'
    wptt_names = os.listdir(root)
    for wptt_name in wptt_names:
        dataset = get_dataset(root, wptt_name)
        for line_idx in range(dataset[1]):
            with open('dataset/WPTTtxt/Train/' + wptt_name[:-5] + '_' + str(line_idx) + '.txt', 'w') as file:
                file.write(dataset[2][line_idx][2] + '\n')
                for row in dataset[2][line_idx][3]:
                    for i in row:
                        #  通过join函数转换row为字符串，并以逗号分隔元素
                        file.write(str(i[0]) + ',' + str(i[1]) + '\n')


def plt_print():
    root = 'data/WPTT/WPTT-Train/'
    file_name = '009-P16.wptt'
    data_list = get_dataset(root, file_name)
    for line_idx in range(data_list[1]):
        print(data_list[2][line_idx][2])
        plt.figure(figsize=(20, 2))
        plt.axis('off')
        u = []
        v = []
        for row in data_list[2][line_idx][3]:
            X = []
            Y = []
            for i in row:
                X.append(i[0])
                u.append(i[0])
                Y.append(-i[1])
                v.append(-i[1])
            plt.plot(X, Y, 'k-', linewidth=1)
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
        fx = p1(ax)
        # plt.axis('off')
        # plt.plot(ax, fx, color='green')
        # plt.show()

        X.clear()
        Y.clear()
        u.clear()
        v.clear()
        plt.figure(figsize=(20, 2))
        for row in data_list[2][line_idx][3]:
            X = []
            Y = []
            for i in row:
                X.append(i[0])
                u.append(i[0])
                Y.append(-i[1]-(p1(i[0])-min(ax)))
                v.append(-i[1]-(p1(i[0])-min(ax)))
            plt.axis('off')
            plt.plot(X, Y, 'k-', linewidth=1)
        plt.show()
        avex2 = []
        avey2 = []
        s2 = 0
        sux2 = 0
        suy2 = 0
        for i in range(len(u)):
            s2 += 1
            if s2 == 4:
                avex2.append(sux2 / 4)
                avey2.append(suy2 / 4)
                s2 = 0
                sux2 = 0
                suy2 = 0
            sux2 += u[i]
            suy2 += v[i]
        ax2 = np.array(avex2)
        ay2 = np.array(avey2)
        z2 = np.polyfit(ax2, ay2, 2)
        p2 = np.poly1d(z2)
        fx2 = p2(ax2)
        # plt.axis('off')
        # plt.plot(ax2, fx2, color='green')
        # plt.show()


if __name__ == '__main__':
    plt_print()
