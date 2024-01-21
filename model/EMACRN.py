import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_size: int = None,
                 embedding_size: int = None,
                 multihead_num: int = None):  # drop_rate: float = None
        super(MultiHeadAttention, self).__init__()
        self.source_size = input_size
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        # self.drop_rate = drop_rate

        self.linear = nn.Linear(in_features=self.embedding_size, out_features=self.source_size)
        self.linear_q = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)
        self.linear_k = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)
        self.linear_v = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.source_size)

    def forward(self, inputs, mask=None):
        assert isinstance(inputs, list)
        q = inputs[0]
        k = inputs[1]
        v = inputs[-1] if len(inputs) == 3 else k
        batch_size = q.size(0)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = torch.cat(torch.split(q, split_size_or_sections=self.embedding_size // self.multihead_num, dim=-1), dim=0)
        k = torch.cat(torch.split(k, split_size_or_sections=self.embedding_size // self.multihead_num, dim=-1), dim=0)
        v = torch.cat(torch.split(v, split_size_or_sections=self.embedding_size // self.multihead_num, dim=-1), dim=0)

        attention = torch.matmul(q, k.transpose(2, 1)) / torch.sqrt(torch.tensor(self.embedding_size
                                                                                 // self.multihead_num).float())

        if mask is not None:
            # mask = mask.repeat(self.multihead_num, 1, 1)
            attention -= 1e+9 * mask
        attention = torch.softmax(attention, dim=-1)

        feature = torch.matmul(attention, v)
        feature = torch.cat(torch.split(feature, split_size_or_sections=batch_size, dim=0), dim=-1)
        output = self.linear(feature)
        # output = torch.dropout(output, p=self.drop_rate, train=True)

        output = torch.add(output, inputs[0])
        output = self.layer_norm(output)

        return output, attention


class BLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.f1 = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        out_put = self.f1(t_rec)
        out_put = out_put.view(T, b, -1)
        return out_put


class BasicConv2d1x1(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, **kwargs):
        super(BasicConv2d, self).__init__()
        self.padding = nn.ZeroPad2d((dilation_rate, dilation_rate, 0, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x


class ACM1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACM1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=2, padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Mish()
                                   )
        self.conv2 = nn.Sequential(nn.MaxPool2d((1, 2), 2),
                                   nn.ZeroPad2d((1, 1, 0, 0)),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=1, padding=0,
                                             groups=2),
                                   nn.ZeroPad2d((1, 1, 0, 0)),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=0,
                                             groups=2),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Mish()
                                   )
        self.conv1x1 = BasicConv2d1x1(out_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        output = torch.cat([x1, x2], 1)
        output = self.conv1x1(output)
        return output


class ACM2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACM2, self).__init__()
        self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 1, 0, 0)),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=1, padding=0,
                                             groups=2),
                                   nn.ZeroPad2d((1, 1, 0, 0)),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=0,
                                             groups=2),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Mish()
                                   )
        self.conv1x1 = BasicConv2d1x1(out_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        output = torch.cat([x, x1], 1)
        output = self.conv1x1(output)
        return output


class EPAM1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EPAM1, self).__init__()
        self.conv1 = BasicConv2d(in_channels, out_channels, 1, kernel_size=(2, 3), stride=1, padding=0, dilation=(1, 1))
        self.conv2 = BasicConv2d(in_channels, out_channels, 2, kernel_size=(2, 3), stride=1, padding=0, dilation=(1, 2))
        self.conv3 = BasicConv2d(in_channels, out_channels, 3, kernel_size=(2, 3), stride=1, padding=0, dilation=(1, 3))
        self.conv4 = BasicConv2d(in_channels, out_channels, 5, kernel_size=(2, 3), stride=1, padding=0, dilation=(1, 5))
        self.conv1x1_2 = BasicConv2d1x1(out_channels * 4, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], 1)
        out = self.conv1x1_2(out)
        return out

class EPAM2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EPAM2, self).__init__()
        self.conv1 = BasicConv2d(in_channels, out_channels, 1, kernel_size=(1, 3), stride=1, padding=0, dilation=1)
        self.conv2 = BasicConv2d(in_channels, out_channels, 2, kernel_size=(1, 3), stride=1, padding=0, dilation=2)
        self.conv3 = BasicConv2d(in_channels, out_channels, 3, kernel_size=(1, 3), stride=1, padding=0, dilation=3)
        self.conv4 = BasicConv2d(in_channels, out_channels, 5, kernel_size=(1, 3), stride=1, padding=0, dilation=5)
        self.conv1x1_2 = BasicConv2d1x1(out_channels * 4, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], 1)
        out = self.conv1x1_2(out)
        return out

class ATM(nn.Module):
    def __init__(self):
        super(ATM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1= nn.Conv1d(1, 256,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv1d(256,1,kernel_size=1,stride=1)
    def forward(self,x):
        y = x
        x = self.avgpool(x)
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.softmax(x,dim=-1)
        x = x.permute(0,2,1)
        output = x*y+y
        return output

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool1d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        print(out.shape)
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1)
        return out*x


class End2end(nn.Module):
    def __init__(self, nclass, nh):
        super(End2end, self).__init__()

        self.pad = nn.ZeroPad2d((0, 1, 0, 0)) # iahwdb
        # self.pad = nn.ZeroPad2d((0, 3, 0, 0))  # casia 1980
        # self.pad = nn.ZeroPad2d((0, 0, 0, 0))  # casia 2247
        self.conv1 = EPAM1(1, 64)
        self.conv2 = EPAM2(64, 128)
        self.conv3 = EPAM2(128, 256)

        self.acm11 = ACM1(256, 256)
        self.acm21 = ACM2(256, 256)
        self.acm12 = ACM1(256, 256)
        self.acm22 = ACM2(256, 256)
        self.acm13 = ACM1(256, 256)
        self.acm23 = ACM2(256, 256)

        self.att1 = MultiHeadAttention(256, 256, 4)
        self.att2 = MultiHeadAttention(256, 256, 4)

        self.lstmg1 = BLSTM(256, nh, nh)
        self.lstmg2 = BLSTM(nh, nh, 256)
        self.fc = nn.Linear(256, nclass)
        # self.atm= ATM()
        # self.eca=ECA(256)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.acm11(x)
        x = self.acm21(x)
        x = self.acm12(x)
        x = self.acm22(x)
        x = self.acm13(x)
        x = self.acm23(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = x.permute(2, 0, 1)
        x = self.lstmg1(x)
        x = self.lstmg2(x)
        conv1 = x.permute(1, 0, 2)
        # output = self.eca(conv1)
        # output = self.atm(conv1)
        output, _ = self.att1([conv1, conv1, conv1])
        output, _ = self.att2([output, output, output])

        output = self.fc(output)
        output = output.permute(1, 0, 2)

        return F.log_softmax(output, dim=2)


if __name__ == '__main__':
    a = torch.randn(32, 1, 2, 2247)
    net = End2end(3999, 320)
    c = net(a)
    print(c.shape)


