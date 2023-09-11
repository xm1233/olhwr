import random
import numpy as np
import torch
from PyQt5.QtGui import QPixmap
from matplotlib import pyplot as plt
import prepro
from model.eacrn import End2end
from utils.chinese_char import chars_list2, chars_list_casia
from utils.iutils import strLabelConverter


def load_model_recognition(label_img, label_text, load_model):
    # self.timer_camera.stop()  # 关闭定时器
    # self.button_open_camera.setText('打开相机')
    X = []
    Y = []
    with open('./output_file/xy.txt', 'r') as f:
        for l in f.readlines():
            X.append(float(l.split(',')[0].strip('\n')))
            Y.append(float(l.split(',')[1].strip('\n')))
    q, v = prepro.normLization_11(X, Y)
    q, v = prepro.trajectory_preprocessing(q, v, 0.02)
    with open('./output_file/text_feature.txt', 'w') as f:
        for i in range(len(q)):
            f.write(str(q[i]) + ',' + str(v[i]) + '\n')
    with open('./output_file/text_xy.txt', 'w') as f:
        for i in range(len(q)):
            f.write(str(q[i]) + ',' + str(v[i]) + '\n')
    prepro.Max_feature_padding('./output_file/text_feature.txt')
    net = End2end(len(chars_list2) + 1, 256)
    params = torch.load(load_model, map_location='cpu')
    net.load_state_dict(params)
    net.eval()
    m = []
    trajx = []
    trajy = []
    with open('./output_file/text_xy.txt', 'r') as fw:
        for line in fw.readlines():
            trajx.append(float(line.split(',')[0]))
            trajy.append(float(line.split(',')[1].strip('\n')))
    width = np.max(trajx) - np.min(trajx)
    height = np.max(trajy) - np.min(trajy)
    nw = width / height * 2
    plt.figure(figsize=(nw * 2, 2))
    current_axes = plt.axes()
    current_axes.xaxis.set_visible(False)
    current_axes.yaxis.set_visible(False)
    plt.axis('off')
    plt.plot(trajx, trajy)
    plt.savefig('./output_file/1.png')
    pix = QPixmap('./output_file/1.png')
    label_img.setPixmap(pix)
    label_img.setScaledContents(True)
    with open('./output_file/text_feature.txt', "r") as fp:
        for l in fp.readlines():
            l = l.strip('\n')
            m.append(l.split(','))
    x = torch.Tensor(np.array(m).astype(float)).permute(1, 0)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    convert = strLabelConverter("".join(chars_list2))
    out = net(x)
    _, pres = out.max(2)
    pre = pres.transpose(1, 0).contiguous().view(-1)
    pre_len = torch.IntTensor([out.size(0)] * out.shape[1])
    sim_pre_raw = convert.decode(pre, pre_len, raw=True)
    sim_pre = convert.decode(pre, pre_len, raw=False)
    label_text.setText("预测文本行:  " + sim_pre.strip())