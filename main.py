import sys
import timm.scheduler

from plt_show import plt_show

sys.path.append('D:\\pyproject\\TCRNdemo\\utils')
import torch
import torch.nn as nn
import torch.optim as optim
from model.tcrn import TCRN, weights_init1, setup_seed, Dilation_conv1_res, Dilation_conv1_2, weights_init2
from model.e2e_gap import End2end  # from model.eacrn import End2end
from utils.iutils import strLabelConverter, str_dsitance
from model.focal_ctc_loss import focal_loss
from utils.text_dataset import Mydataset, Mydataset_end2end
from utils.chinese_char import chars_list2, chars_list_casia
import torch.utils.data as data
import os
import timm
import time


if __name__ == '__main__':
    setup_seed(4)
    torch.backends.cudnn.enabled = False

    train_dataset = Mydataset(chars_list_casia, './data/txt/6d_feature_casia/train_data.txt', train='train')
    train_dataset_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # val_dataset = Mydataset_end2end(chars_list2, './data/txt/6d_feature_casia/test_data.txt', train='val')
    # val_dataset_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)

    # test_dataset = Mydataset(chars_list2, './data/txt/old_casia_2247/cmp13_data.txt', train='test')
    # test_dataset_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

    convert = strLabelConverter("".join(chars_list_casia))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = TCRN(len(chars_list_casia) + 1, 200).to(device)
    weight_p = []
    bias_p = []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    # params = torch.load('./pre-model/net_22_aracc_0.7081_cracc0.7191.pth')
    # params = torch.load('./pre_model/checkpoint_49_aracc_0.9603_cracc0.9655.pth')
    # net.load_state_dict(params)
    # net.load_state_dict(params['state_dict'])
    net.to(device)
    # net.apply(weights_init1)
    critrion = nn.CTCLoss()
    # critrion = focal_loss()
    optimzier = optim.Adam(net.parameters(), lr=0.001)
    # optimzier = optim.Adam([{'params': weight_p, 'weight_dacay': 1e-5}, {'params': bias_p, 'weight_decay': 0}],
    # lr=0.0001) scheduler = timm.scheduler.CosineLRScheduler(optimzier, 200, 1e-5, 10, 1e-5)

    # scheduler = optim.lr_scheduler.StepLR(optimzier, step_size=3, gamma=0.1)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimzier, gamma=0.98)

    for epoch in range(200):
        # scheduler.step(epoch)
        train_loss = 0.0
        AR_corrects = 0
        CR_corrects = 0
        N = 0
        train_num = 0
        net.train()
        for step, (x, label) in enumerate(train_dataset_loader):
            x = x.reshape(-1, 6, 2247).to(device)
            # start = time.time()
            out = net(x)
            # end = time.time()
            # print(end - start)
            _, pres = out.max(2)
            pre = pres.transpose(1, 0).contiguous().view(-1)
            text, lengths = convert.encode(label)
            pre_len = torch.IntTensor([out.size(0)] * out.shape[1])
            loss = critrion(out, text, pre_len, lengths)
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()
            sim_pre = convert.decode(pre, pre_len, raw=False)
            for p, l in zip(sim_pre, label):
                AR_corrects += (len(l) - str_dsitance(l, p, "ar"))
                CR_corrects += (len(l) - str_dsitance(l, p, "cr"))
                N += len(l)
            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)
        print("Epoch {}/{}  train_loss: {:.8f}, AR_acc: {:.8f}, CR_acc: {:.8f}".format(epoch + 1, 100,
                                                                                       train_loss / train_num,
                                                                                       AR_corrects / N,
                                                                                       CR_corrects / N))

        # scheduler.step()
        # net.eval()
        # val_loss = 0.0
        # val_num = 0
        # AR_corrects = 0
        # CR_corrects = 0
        # N = 0
        # with torch.no_grad():
        #     for s, (x, label) in enumerate(val_dataset_loader):
        #         x = x.reshape(-1, 1, 2, 2998).to(device)
        #         out = net(x)
        #         # print(out.shape)
        #         _, pres = out.max(2)
        #         pre = pres.transpose(1, 0).contiguous().view(-1)
        #         text, lengths = convert.encode(label)
        #         pre_len = torch.IntTensor([out.size(0)] * out.shape[1])
        #         loss = critrion(out, text, pre_len, lengths)
        #         sim_pre = convert.decode(pre, pre_len, raw=False)
        #         for p, l in zip(sim_pre, label):
        #             AR_corrects += (len(l) - str_dsitance(l, p, "ar"))
        #             CR_corrects += (len(l) - str_dsitance(l, p, "cr"))
        #             N += len(l)
        #         val_loss += loss.item() * x.size(0)
        #         val_num += x.size(0)
        # print("               val_loss: {:.8f}, AR_acc: {:.8f}, CR_acc: {:.8f}".format(val_loss / val_num,
        #                                                                                AR_corrects / N,
        #                                                                                CR_corrects / N))
        #
        # torch.save(net.state_dict(), os.path.join('./output/', "net_{}_aracc_{:.4f}_cracc{:.4f}.pth".format(epoch,
        #                                                                                                     AR_corrects / N,
        #                                                                                                     CR_corrects / N)))
        #
        # with open('./data/val7.txt', 'a+') as vf:
        #     vf.write(str(epoch + 1) + "," + str(val_loss / val_num) + "," + str(AR_corrects / N) + "," + str(
        #         CR_corrects / N) + '\n')

    #     net.eval()
    #     test_loss = 0.0
    #     test_num = 0
    #     AR_corrects = 0
    #     CR_corrects = 0
    #     N = 0
    #     with torch.no_grad():
    #         for s, (x, label) in enumerate(test_dataset_loader):
    #             x = x.reshape(-1, 6, 2247).to(device)
    #             out = net(x)
    #             # print(out.shape)
    #             _, pres = out.max(2)
    #             pre = pres.transpose(1, 0).contiguous().view(-1)
    #             text, lengths = convert.encode(label)
    #             pre_len = torch.IntTensor([out.size(0)] * out.shape[1])
    #             loss = critrion(out, text, pre_len, lengths)
    #             sim_pre = convert.decode(pre, pre_len, raw=False)
    #             for p, l in zip(sim_pre, label):
    #                 p = p.replace(' ', '')
    #                 l = l.replace(' ', '')
    #                 AR_corrects += (len(l) - str_dsitance(l, p, "ar"))
    #                 CR_corrects += (len(l) - str_dsitance(l, p, "cr"))
    #                 N += len(l)
    #             test_loss += loss.item() * x.size(0)
    #             test_num += x.size(0)
    #     print("              test_loss: {:.8f}, AR_acc: {:.8f}, CR_acc: {:.8f}".format(test_loss / test_num,
    #                                                                                    AR_corrects / N,
    #                                                                                    CR_corrects / N))
    #     print("-----------------------------------------------------------------------------------------------------")
    # plt_show('./data/val7.txt')
