from pycorrector.macbert.macbert_corrector import MacBertCorrector
import kenlm
from utils.iutils import str_dsitance
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
'''f = open('../data/error_test.txt', 'r')
ar = 0
cr = 0
N = 0
for line in f.readlines():
    label = line.split(' ')[0]
    predict = line.split(' ')[1].strip('\n')
    sentence, _ = pycorrector.correct(predict)
    ar += max(len(label) - str_dsitance(label, sentence, "ar"), len(label) - str_dsitance(label, predict, "ar"))
    cr += max(len(label) - str_dsitance(label, sentence, "cr"), len(label) - str_dsitance(label, predict, "cr"))
    N += len(label)
print(N, ar, cr, ar/N, cr/N)'''

s = "个教堂里。从全克方场沿河直向东去，"
m = MacBertCorrector("shibing624/macbert4csc-base-chinese").macbert_correct
print(m(s))

