#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: DeepCold
@file: train.py
@time: 2020/7/29 16:14
"""
import os
import cv2
import joblib
from sklearn.svm import SVC


train_set_x = []
train_set_y = []

path = "./train_img"

# 遍历文件夹 获取下面的目录
for category in os.listdir(path):  # listdir的参数是文件夹的路径
    for dir_name in os.listdir(os.path.join(path, category)):
        for file_name in os.listdir(os.path.join(path, category, dir_name)):
            img1 = cv2.imread(os.path.join(path, category, dir_name, file_name), cv2.IMREAD_GRAYSCALE)
            res1 = cv2.resize(img1, (28, 28))
            res1_1 = res1.reshape(784)  # 将表示图片的二维矩阵转换成一维
            res1_1_1 = res1_1.tolist()  # 将numpy.narray类型的矩阵转换成list
            train_set_x.append(res1_1_1)  # 将list添加到已有的list中
            train_set_y.append(dir_name)

letterSVM = SVC(kernel="linear", C=1).fit(train_set_x, train_set_y)
# 生成训练结果
joblib.dump(letterSVM, './model_data/letter.pkl')
