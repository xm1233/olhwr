import numpy as np
import torch.utils.data as data
import torch
import os
from utils.chinese_char import chars_list2
from utils.iutils import strLabelConverter
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''with open("./train.txt", "a+") as f:
    for filename in os.listdir("./txt/"):
        f.write(filename+'\n')'''
# a = []
# b = []
'''with open("./train.txt", "r") as f:
    for line in f.readlines():
        a.append(str(line).strip('\n'))
with open("./4.txt", "r") as f:
    for line in f.readlines():
        b.append(str(line).strip('\n'))
with open("./all.txt", "a+") as f:
    for i in range(len(a)):
        f.write(a[i]+","+b[i]+"\n")'''


class Mydataset(data.Dataset):
    def __init__(self, char_list, file_path, train, transform=None):
        self.char_list = char_list
        self.char2index = {self.char_list[i]: i for i in range(len(self.char_list))}
        self.paths = file_path
        self.f = open(file_path, encoding='utf-8')
        self.transform = transform
        self.train = train
        self.paths = []
        self.labels = []

        for line in self.f.readlines():
            self.paths.append(line.split('$')[0])
            self.labels.append(line.split('$')[1])

    def __getitem__(self, item):
        m = []
        if self.train == 'train':
            with open('./data/txt/6d_feature_casia/Train_feature_0/'+self.paths[item], "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    m.append(line.split(' '))
            x = torch.Tensor(np.array(m, dtype='float32')).permute(1, 0)
        elif self.train == 'val':
            with open('./data/txt/6d_feature_casia/Test_feature_0/'+self.paths[item], "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    m.append(line.split(' '))
            x = torch.Tensor(np.array(m, dtype='float32')).permute(1, 0)
        else:
            with open('./data/txt/6d_feature_casia/cmp13_feature_0/'+self.paths[item], "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    m.append(line.split(' '))
            x = torch.Tensor(np.array(m, dtype='float32')).permute(1, 0)

        label = self.labels[item].strip('\n')
        # label = self.numerical(self.labels[item])
        return x, label

    def __len__(self):
        return len(self.paths)

    def numerical(self, chars):
        char_tensor = torch.zeros(len(chars))
        for i in range(len(chars)):
            char_tensor[i] = self.char2index[chars[i]]+1
        return char_tensor


class Mydataset_end2end(data.Dataset):
    def __init__(self, char_list, file_path, train, transform=None):
        self.char_list = char_list
        self.char2index = {self.char_list[i]: i for i in range(len(self.char_list))}
        self.paths = file_path
        self.f = open(file_path, encoding='utf-8')
        self.transform = transform
        self.train = train
        self.paths = []
        self.labels = []

        for line in self.f.readlines():
            self.paths.append(line.split('$')[0])
            self.labels.append(line.split('$')[1])

    def __getitem__(self, item):
        m = []
        if self.train == 'train':
            with open('./data/txt/old_casia_2247/casia_e2e_train_feature/'+self.paths[item], "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    m.append(line.split(','))
            x = torch.Tensor(np.array(m, dtype='float64')).permute(1, 0)
        elif self.train == 'val':
            with open('./data/txt/old_casia_2247/casia_e2e_test_feature'+self.paths[item], "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    m.append(line.split(','))
            x = torch.Tensor(np.array(m, dtype='float64')).permute(1, 0)
        else:
            with open('./data/txt/old_casia_2247/cmp13_e2e_feature/'+self.paths[item], "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    m.append(line.split(','))
            x = torch.Tensor(np.array(m, dtype='float64')).permute(1, 0)

        label = self.labels[item].strip('\n')
        return x, label

    def __len__(self):
        return len(self.paths)

    def numerical(self, chars):
        char_tensor = torch.zeros(len(chars))
        for i in range(len(chars)):
            char_tensor[i] = self.char2index[chars[i]]+1
        return char_tensor


class Mydataset_transducer(data.Dataset):
    def __init__(self, char_list, file_path, train, transform=None):
        self.char_list = char_list
        self.char2index = {self.char_list[i]: i for i in range(len(self.char_list))}
        self.paths = file_path
        self.f = open(file_path, encoding='utf-8')
        self.transform = transform
        self.train = train
        self.paths = []
        self.labels = []

        for line in self.f.readlines():
            self.paths.append(line.split('$')[0])
            self.labels.append(line.split('$')[1])

    def __getitem__(self, item):
        m = []
        if self.train == 'train':
            with open('./data/txt/s_train_feature/'+self.paths[item], "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    m.append(line.split(' '))
            with open('./data/txt/s_train_feature/' + self.paths[item], "r") as f:
                f_loss = len(f.readlines())
            with open('./data/txt/s_train_xy/'+self.paths[item], 'r') as fl:
                feature_len = len(fl.readlines())
            x = torch.Tensor(np.array(m, dtype='float32')).permute(1, 0)
        else:
            with open('./data/txt/s_test_feature/'+self.paths[item], "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    m.append(line.split(' '))
            with open('./data/txt/s_test_xy/'+self.paths[item], 'r') as fl:
                feature_len = len(fl.readlines())
            x = torch.Tensor(np.array(m, dtype='float32')).permute(1, 0)

        label = self.labels[item].strip('\n')
        t_label = self.labels[item].replace('#', '').replace('￥', '').strip('\n')
        l_length = len(self.labels[item].replace('￥', '').strip().strip('\n'))
        # label = self.numerical(self.labels[item])
        return x, label, t_label, feature_len, l_length, f_loss

    def __len__(self):
        return len(self.paths)

    def numerical(self, chars):
        char_tensor = torch.zeros(len(chars))
        for i in range(len(chars)):
            char_tensor[i] = self.char2index[chars[i]]+1
        return char_tensor


if __name__ == '__main__':
    train_dataset = Mydataset_end2end(chars_list2, '../data/txt/IAHWDB_train_data.txt', train='train')
    train_dataset_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    convert = strLabelConverter("".join(chars_list2))
    for i, (s, label) in enumerate(train_dataset_loader):
        text, length = convert.encode(label)
        print(text, length)
        print(text.shape, length.shape)









