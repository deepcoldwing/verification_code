#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: DeepCold
@file: split_image.py
@time: 2020/7/28 15:13
"""
import os
import cv2
from matplotlib import pyplot as plt, cm

from utils.deal_image import noise_remove_cv2, cut_vertical

"""图片的预处理及切割"""


def show_gray_img(img):
    plt.imshow(img, cmap=cm.gray)
    plt.show()


if __name__ == '__main__':
    source_root = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(source_root, "image_raw")  # 原图片文件夹
    new_image_path = os.path.join(source_root, "image_after_split")  # 切割好的图片文件夹
    files = os.listdir(image_path)
    for key, file in enumerate(files):
        img = cv2.imread(os.path.join(image_path, file))
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        # 二值化处理(根据具体图片调整阈值参数，也就是第二个参数)
        ret, im_inv = cv2.threshold(im_gray, 140, 255, 0)
        img_clear = noise_remove_cv2(im_inv, 1)  # 去除孤立点，噪点
        show_gray_img(img_clear)
        img_list = cut_vertical(img_clear)  # 垂直分割投影法分割图片
        t = 1
        for i in img_list:
            resize_img = cv2.resize(i, (15, 30))  # 重新定义大小
            # 这里可以对切割到的图片进行操作，显示出来或者保存下来
            cv2.imwrite(os.path.join(new_image_path, file.split(".")[0] + "_" + str(t) + '.jpg'), resize_img)
            t += 1
