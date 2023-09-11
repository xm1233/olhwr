import sys
import random
import time

sys.path.append('D:\\pyproject\\TCRNdemo\\utils')
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model.tcrn import TCRN, weights_init1, setup_seed, Dilation_conv1_res, Dilation_conv1_2, weights_init2
from model.Lstm_model import layer_lstm
from model.eacrn import End2end
from utils.iutils import strLabelConverter, str_dsitance
from utils.text_dataset import Mydataset
from utils.chinese_char import chars_list2, chars_list_casia
import torch.utils.data as data
import os
import numpy as np

net = End2end(len(chars_list_casia)+1, 256)
params = torch.load('D:\\pyproject\\TCRNdemo\\pre-model\\cmp13_2247\\ctc_aracc_0.8732_cracc0.8781.pth')
net.load_state_dict(params)
net.eval()
m = []
line = random.randint(0, 3000)
print(line)
trajx = []
trajy = []
with open('./data/txt/old_casia_2247/cmp13_data.txt', "r", encoding='utf-8') as f:
    y = f.readlines()[line].split('$')
    lab = y[1].strip('\n')
    da = y[0]
    with open('./data/txt/old_casia_2247/cmp13_e2e_feature/'+da, 'r') as fw:
        for line in fw.readlines():
            trajx.append(float(line.split(',')[0]))
            trajy.append(float(line.split(',')[1].strip('\n')))
    # width = np.max(trajx) - np.min(trajx)
    # height = np.max(trajy) - np.min(trajy)
    # nw = width / height * 2
    # plt.figure(figsize=(nw, 1), frameon=False)
    # current_axes = plt.axes()
    # current_axes.xaxis.set_visible(False)
    # current_axes.yaxis.set_visible(False)
    # plt.axis('off')
    # plt.plot(trajx, trajy)
    # plt.savefig("1.jpg", dpi=600, bbox_inches='tight')
    # plt.show()
    with open('./data/txt/old_casia_2247/cmp13_e2e_feature/'+da, "r") as fp:
        for l in fp.readlines():
            l = l.strip('\n')
            m.append(l.split(','))
x = torch.Tensor(np.array(m).astype(float)).permute(1, 0)
x = x.unsqueeze(0)
x = x.unsqueeze(0)


convert = strLabelConverter("".join(chars_list_casia))
# start = time.time()
out = net(x)
# end = time.time()
# print(end-start)
_, pres = out.max(2)
# number, p = out.topk(3, dim=2)
# print(number)

pre = pres.transpose(1, 0).contiguous().view(-1)
pre_len = torch.IntTensor([out.size(0)] * out.shape[1])
sim_pre_raw = convert.decode(pre, pre_len, raw=True)
sim_pre = convert.decode(pre, pre_len, raw=False)
# p = p[:, :, 1]
# p = p.transpose(1, 0).contiguous().view(-1)
# p_len = torch.IntTensor([out.size(0)]*out.shape[1])
# sim_p = convert.decode(p, p_len, raw=True)

# print(sim_p.strip())
# print(sim_pre_raw.strip())
print("真实文本行:  "+lab.strip())
print("预测文本行:  "+sim_pre.strip())


