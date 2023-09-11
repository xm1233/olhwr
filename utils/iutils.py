import torch
from utils.chinese_char import chars_list1, chars_list2


# 评估函数
def str_dsitance(label, pre, flag):
    n = len(label)
    m = len(pre)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = dp[i - 1][j] + 1
            if flag == "ar":
                right = dp[i][j - 1] + 1
            else:
                right = dp[i][j - 1]
            l_r = dp[i - 1][j - 1]
            if label[i - 1] != pre[j - 1]:
                l_r += 1
            dp[i][j] = min(left, right, l_r)
    return dp[-1][-1]


# encode&&decode
class strLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet + '@'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, text):

        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False

        for item in text:
            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                result.append(self.dict[char])
        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):

        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


# 获取字典
def get_char_dict(path):
    # with open(path, 'rb') as file:
    #     char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}

    with open(path, 'r', encoding='utf-8') as file:
        char_dict = {num: char.strip() for num, char in enumerate(file.readlines())}


if __name__ == "__main__":
    # a = '地从水中吸取它所需要的元素水'
    # b = '地从水中吸取它所需要的元素水'
    # print(str_dsitance(a, b, 'cr'))
    con = strLabelConverter("".join(chars_list2))
    a = [23, 56, 23, 21, 231]
    b = torch.tensor(a)