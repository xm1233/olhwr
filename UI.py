from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2.cv2 as cv2
import os
import torch
import torch.nn as nn
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QBrush
from PyQt5.QtWidgets import QFileDialog
from sklearn.metrics import accuracy_score
from UiUtils.load_model_recognition import load_model_recognition
from UiUtils.paint_point import paint_points
from UiUtils.computer_simulated_drawing import ChildWindow, computer_simulated
from main import start_train

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def change_asc2(a):
    if 0 <= a.item() <= 9:
        return chr(a.item() + 48)
    elif a.item() % 2 == 0:
        return chr(int((a.item() - 10) / 2) + 97)
    elif a.item() % 2 != 0:
        return chr(int((a.item() - 11) / 2) + 65)


def change_char(a):
    file = open('./data.txt', 'r', encoding='utf-8')
    keys = []
    value = []
    for l in file.readlines():
        keys.append(int(l.split('\t')[0]))
        value.append(l.split('\t')[1])
    dict1 = dict(zip(keys, value))
    return dict1[a.item() + 1]


class Status(object):
    def __init__(self):
        self.__status = 0

    def getStatus(self):
        return self.__status

    def setStatus(self, value):
        if isinstance(value, int):
            self.__status = value


def Simulated_drawing():
    childwindow = ChildWindow()
    f = open('./output_file/xy.txt', 'w').close()
    childwindow.exec()
    # computer_simulated(self.status)


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0 + cv2.CAP_DSHOW  # 为0时表示视频流来自笔记本内置摄像头，不加cv2.CAP_DSHOW会提示显存不够
        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.outname = 'D:\\pyproject\\TCRNdemo\\pre-model\\e2e_group\\eacrn_group2_aracc_0.8430_cracc0.8559.pth'
        self.status = Status()

    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QVBoxLayout()  # 总布局
        self.__layout_fun_button = QtWidgets.QHBoxLayout()  # 按键布局
        self.__layout_show = QtWidgets.QHBoxLayout()  # label显示布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 识别图片与结果显示
        self.button_open_camera = QtWidgets.QPushButton('打开相机')  # 建立用于打开摄像头的按键
        self.button_file = QtWidgets.QPushButton('选择模型')
        self.button_train = QtWidgets.QPushButton('训练模型')
        self.button_simulated_drawing = QtWidgets.QPushButton('模拟书写')
        self.button_begin_recognize = QtWidgets.QPushButton('开始识别')  # 建立用于开始识别的按键
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_open_camera.setMinimumHeight(60)  # 设置按键大小
        self.button_file.setMinimumHeight(60)
        self.button_train.setMinimumHeight(60)
        self.button_simulated_drawing.setMinimumHeight(60)
        self.button_begin_recognize.setMinimumHeight(60)  # 设置按键大小
        self.button_close.setMinimumHeight(60)
        # self.button_close.move(20, 100)  # 移动按键
        '''信息显示'''
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_img = QtWidgets.QLabel()  # 定义img的Label
        self.label_show_recognize = QtWidgets.QLabel()  # 定义显示识别的Label
        self.label_show_recognize.setFont(QFont('Times', 14))
        self.label_show_recognize.setFrameShape(QtWidgets.QFrame.Box)
        self.label_show_recognize.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_show_recognize.setFrameShape(QtWidgets.QFrame.Box)
        self.label_show_camera.setFixedSize(960, 780)  # 给显示视频的Label设置大小为560*560
        self.label_show_img.setFixedSize(640, 100)  # 给显示图片的Label设置大小为560*560
        self.label_show_recognize.setFixedSize(640, 300)
        '''把按键加入到按键布局中'''

        self.__layout_fun_button.addWidget(self.button_train)  # 将开始训练的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_train)  # 将开始训练的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_file)  # 把选择模型的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_simulated_drawing)  # 把模拟书写的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_begin_recognize)  # 把开始识别的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中

        '''把某些控件加入到总布局中'''

        self.__layout_show.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        self.__layout_data_show.addWidget(self.label_show_img)
        self.__layout_data_show.addWidget(self.label_show_recognize)
        self.__layout_show.addLayout(self.__layout_data_show)
        self.__layout_main.addLayout(self.__layout_show)
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_data_show.setSpacing(50)
        self.__layout_fun_button.setSpacing(50)
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)
        self.button_simulated_drawing.clicked.connect(Simulated_drawing)
        self.button_file.clicked.connect(self.choose_file)
        self.button_begin_recognize.clicked.connect(
            self.button_begin_recognize_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

    '''槽函数之一'''

    def start_train(self):
        start_train()

    def choose_file(self):
        filename = QFileDialog.getOpenFileNames(self, '选择模型', os.getcwd(), "All Files(*);;Text Files(*.txt)")
        print(filename[0][0])
        if len(filename[0]):
            self.outname = filename[0][0]

    def button_open_camera_clicked(self):
        if not self.timer_camera.isActive():  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if not flag:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开相机')

    def show_camera(self):
        paint_points(self.cap, self.label_show_camera)

    def button_begin_recognize_clicked(self):
        load_model_recognition(self.label_show_img, self.label_show_recognize, self.outname)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.resize(1920, 1080)
    ui.setWindowTitle("手写文本识别系统v1.0")
    palette = QPalette()
    pix = QPixmap("./output_file/background.jpg")
    palette.setBrush(QPalette.Background, QBrush(pix))
    ui.setPalette(palette)
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
