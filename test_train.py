#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: DeepCold
@file: test_train.py
@time: 2020/7/29 17:19
"""

import cv2
import joblib

from split_image import noise_remove_cv2, cut_vertical


def ocr_img(file_name):
    captcha = []
    clf = joblib.load('model_data/letter.pkl')
    img = cv2.imread(file_name)
    # 转换为灰度图
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化处理
    ret, im_inv = cv2.threshold(im_gray, 140, 255, 0)
    # 去除孤立点，噪点
    img_clear = noise_remove_cv2(im_inv, 1)
    # 垂直分割投影法分割图片
    img_list = cut_vertical(img_clear)
    for i in img_list:
        res1 = cv2.resize(i, (28, 28))
        data = res1.reshape(784)
        data = data.reshape(1, -1)

        one_letter = clf.predict(data)[0]
        # print(oneLetter)
        captcha.append(one_letter)
    captcha = [str(i) for i in captcha]
    print("the captcha is :{}".format("".join(captcha)))


if __name__ == '__main__':
    ocr_img("./test_img/test_img_1.png")
