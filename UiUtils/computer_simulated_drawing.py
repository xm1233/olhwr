import sys
import time
import keyboard
from PyQt5.QtCore import QSize, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QPainterPath
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog, QApplication
from matplotlib import pyplot as plt
import pyautogui as pag
import prepro


def insert_x_y(x, y):
    with open('./output_file/xy.txt', 'a+') as f:
        f.write(str(x) + ',' + str(y) + '\n')


def computer_simulated(status):
    X = []
    Y = []
    status.setStatus(0)
    keyboard.add_hotkey('esc', lambda: status.setStatus(0))
    keyboard.add_hotkey('space', lambda: status.setStatus(1))
    flag = 1
    while flag == 1:
        while status.getStatus() == 1:
            x, y = pag.position()
            X.append(x)
            Y.append(-y)
            # 每个30ms中打印一次 , 并执行清屏
            time.sleep(0.01)
        if len(X) > 10:
            status.setStatus(0)
            flag = 0
            q, v = prepro.normLization(X, Y)
            q, v = prepro.trajectory_preprocessing(q, v, 0.00001)
            with open('./output_file/text_feature.txt', 'w') as f:
                for i in range(len(q)):
                    f.write(str(q[i]) + ',' + str(v[i]) + '\n')
            with open('./output_file/text_xy.txt', 'w') as f:
                for i in range(len(q)):
                    f.write(str(q[i]) + ',' + str(v[i]) + '\n')
            prepro.Max_feature_padding('./output_file/text_feature.txt')


class Drawer(QWidget):
    newPoint = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.path = QPainterPath()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPath(self.path)

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        self.newPoint.emit(event.pos())
        self.update()

    def sizeHint(self):
        return QSize(2180, 1080)


class ChildWindow(QDialog):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        keyboard.add_hotkey('esc', lambda: drawer.close())
        self.setLayout(QVBoxLayout())
        self.setWindowTitle('鼠标模拟输入')
        # label = QLabel(self)
        drawer = Drawer(self)
        drawer.newPoint.connect(lambda p: insert_x_y(p.x(), -p.y()))
        # self.layout().addWidget(label)
        self.layout().addWidget(drawer)
