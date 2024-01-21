import torch.nn as nn
import torch
import torch.nn.functional as F


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

        q = torch.cat(torch.split(q, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)
        k = torch.cat(torch.split(k, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)
        v = torch.cat(torch.split(v, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)

        attention = torch.matmul(q, k.transpose(2, 1))/torch.sqrt(torch.tensor(self.embedding_size
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
        self.f1 = nn.Linear(nHidden*2, nOut)
        # self.w_omega = nn.Parameter(torch.Tensor(nHidden*2, nHidden*2))
        # self.u_omega = nn.Parameter(torch.Tensor(nHidden*2, 1))
        # nn.init.uniform_(self.w_omega, 0, 1)
        # nn.init.uniform_(self.u_omega, 0, 1)

    def attn(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        attn_score = F.log_softmax(att, dim=1)
        score_x = x * attn_score
        return score_x

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        # recurrent = recurrent.permute(1, 0, 2)
        # attn_out = self.attn(recurrent)
        # b, T, h = attn_out.size()
        # t_rec = attn_out.reshape(b * T, h)
        # out_put = self.f1(t_rec)
        # out_put = out_put.reshape(T, b, -1)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)
        out_put = self.f1(t_rec)
        out_put = out_put.view(T, b, -1)
        return out_put

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.uniform_(m.weight.data, -1/16, 1/16)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)


class BasicConv2d1x1(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.padding = nn.ZeroPad2d((1, 1, 0, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size=(1, 3), stride=1, padding=0)
        self.conv2 = BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=0)
        self.dropout = nn.Dropout(0.2)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv1x1 = BasicConv2d1x1(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        i = self.conv1x1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + i
        x = self.act2(x)
        return x


class End2end(nn.Module):
    def __init__(self, nclass, nh):
        super(End2end, self).__init__()
        self.conv00 = nn.Conv2d(1, 64, kernel_size=(2, 3), stride=1, padding=0)
        # self.conv01 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=0)
        self.prelu = nn.PReLU()

        self.conv1 = Block(64, 64)
        self.conv2 = Block(64, 64)
        self.conv3 = Block(64, 64)

        self.pool = nn.MaxPool2d((1, 2), 2)

        self.conv4 = Block(64, 128)
        self.conv5 = Block(128, 128)
        self.conv6 = Block(128, 128)

        self.conv7 = Block(128, 256)
        self.conv8 = Block(256, 256)
        self.conv9 = Block(256, 256)

        self.lstmg1 = BLSTM(256, nh, nh)
        self.lstmg2 = BLSTM(nh, nh, 256)
        self.att1 = MultiHeadAttention(256, 256, 8)
        self.att2 = MultiHeadAttention(256, 256, 8)
        self.fc = nn.Linear(256, nclass)

    def forward(self, x):
        x = self.conv00(x)
        # x = self.conv01(x)
        x = self.prelu(x)
        x = self.pool(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.pool(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        b, c, h, w = x.size()
        x = x.view(b, c*h, w)
        x = x.permute(2, 0, 1)
        x = self.lstmg1(x)
        # x = F.interpolate(x, size=[256])
        x = self.lstmg2(x)
        conv1 = x.permute(1, 0, 2)
        output, _ = self.att1([conv1, conv1, conv1])
        output, _ = self.att2([output, output, output])
        output = self.fc(output)
        output = output.permute(1, 0, 2)

        return F.log_softmax(output, dim=2)


if __name__ == '__main__':
    a = torch.randn(32, 1, 2, 2998)
    # a = a.unsqueeze(1)
    # print(a.shape)
    net = End2end(3999, 320)
    print(net(a).shape)