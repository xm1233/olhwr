import math
import torchvision.transforms as transforms
import cv2.cv2 as cv2
import numpy as np
from decimal import *
from PyQt5 import QtCore, QtGui, QtWidgets

dian = []
val_transform = transforms.Compose([
    transforms.ToTensor(),
])


def shujuchuli(a):
    b_x = []
    b_y = []

    for i in range(int(len(a) / 2)):
        b_x.append(a[i * 2])
        b_y.append(a[i * 2 + 1])

    min_value_x = min(b_x)
    max_value_x = max(b_x)

    min_value_y = min(b_y)
    max_value_y = max(b_y)
    new_x = []
    new_y = []
    for i in b_x:
        new_x.append(Decimal(format(((i - min_value_x) / (max_value_x - min_value_x) * 2 - 1), '.6f')).normalize())
    for i in b_y:
        new_y.append(Decimal(format(((i - min_value_y) / (max_value_y - min_value_y) * 2 - 1), '.6f')).normalize())
    new_array = []
    for i in range(len(new_x)):
        new_array.append(new_x[i])
        new_array.append(-new_y[i])
    filename = "D:/pyproject/pythonProject/pythonProject/point_txt/shujuchuli"
    file2 = open(filename + '.txt', 'w')
    file2.write(str(len(new_array)))  # write函数不能写int类型的参数，所以使用str()转化
    file2.write('\n')  # 相当于Tab一下，换一个单元格
    for i in range(len(new_array)):
        file2.write(str(new_array[i]))  # write函数不能写int类型的参数，所以使用str()转化
        file2.write(' ')  # 相当于Tab一下，换一个单元格
        if (i + 1) % 2 == 0:
            file2.write('    \n')  # 写完一行立马换行
    file2.close()


def dist_liangdian(p1, P):
    p1 = np.array(p1)
    p2 = np.array(P)
    p3 = p2 - p1
    p4 = math.hypot(p3[0], p3[1])
    return p4


def paint_points(cap, label_show_camera):
    flag, frame = cap.read()  # 从视频流中读取
    x_i = 0
    frame = cv2.flip(frame, 1)
    # cv2.imshow("video", frame)
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    img_copy = img.copy()
    # 手势提取与二值化
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCrCb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 对图像二值化加自适应阈值
    # 空洞填充
    im_floodfill = skin.copy()
    h, w = skin.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if im_floodfill[i][j] == 0:
                seedPoint = (j, i)
                isbreak = True
                break
        if isbreak:
            break
    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = skin | im_floodfill_inv
    kernel = np.ones((5, 5), np.uint8)
    im_out = cv2.erode(im_out, kernel=kernel, iterations=1)
    # 识别轮廓
    contours, _ = cv2.findContours(im_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # 找到最大的轮廓（根据面积）
            temp = contours[i]
            area = cv2.contourArea(temp)  # 计算轮廓区域面积
            if area > maxArea:
                maxArea = area
                ci = i
        cnt = contours[ci]  # 得出最大的轮廓区域
        mask = np.zeros(im_out.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        imgBin = mask.copy()
        # cv2.imshow("mask", mask)
        # 画最大外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 3)
        # 矩形左上方顶点
        t1 = [x, y]
        # 矩形右下方顶点
        br = [x + w, y + h]
        # 计算到轮廓的距离
        raw_dist = np.empty(img.shape, dtype=np.float32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                raw_dist[i, j] = cv2.pointPolygonTest(contours[0], (j, i), True)
        # 获取最大值即内接圆半径，中心点坐标
        minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
        maxVal = abs(maxVal)
        # 最大内切圆半径
        radius = np.int_(maxVal)
        # 最大内切圆圆心
        center_of_circle = maxDistPt
        P = list(center_of_circle)
        cv2.circle(img, center_of_circle, 8, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, maxDistPt, radius, (0, 0, 255), 1)
        if abs(P[1] - t1[1]) + radius <= br[1]:
            y = int(abs(P[1] - t1[1]) + 1.2 * radius)
            indexPoint = [br[0], y]
            imgBin = cv2.rectangle(imgBin, (t1[0], indexPoint[1]), (br[0], br[1]), (0, 0, 0), -1)
            cv2.circle(imgBin, maxDistPt, int(2 * radius), (0, 0, 0), -1)
            # cv2.imshow("imgBin1", imgBin)
            contours, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for a in range(len(contours)):
                t = []
                x = []
                cnt = contours[a]
                for i in range(len(cnt)):
                    p1 = cnt[i].flatten()
                    x.append(p1)
                    # 计算点到圆心距离
                    dist = dist_liangdian(p1, P)
                    t.append(dist)
                max_index = t.index(max(t))
                x = x[max_index]
                cv2.circle(img_copy, (x[0], x[1]), 2, (0, 0, 255), 6)
                dian.append(x[0])
                dian.append(x[1])
            img_copy = cv2.resize(img_copy, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            # 画手指尖的轨迹
            while True:
                if len(dian) <= 2:
                    break
                cv2.line(img_copy, (dian[x_i] * 2, dian[x_i + 1] * 2), (dian[x_i + 2] * 2, dian[x_i + 3] * 2),
                         (255, 0, 0), 5)
                x_i = x_i + 2
                if x_i + 3 >= len(dian):
                    break
        else:
            img_copy = cv2.resize(img_copy, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    showImage = QtGui.QImage(img_copy.data, img_copy.shape[1], img_copy.shape[0], img_copy.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
    label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里显示QImage
