#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: DeepCold
@time: 2021/11/25 17:17
"""
import numpy as np


# 去除噪点
def noise_remove_cv2(image, k):
    """
    去除图片的噪点
    :param image:
    :param k:
    :return:
    """

    def calculate_noise_count(img_obj, w, h):
        """
        计算邻域非白色的个数
        """
        count = 0
        width, height = img_obj.shape
        for _w_ in [w - 1, w, w + 1]:
            for _h_ in [h - 1, h, h + 1]:
                if _w_ > width - 1:
                    continue
                if _h_ > height - 1:
                    continue
                if _w_ == w and _h_ == h:
                    continue
                if img_obj[_w_, _h_] < 230:  # 二值化的图片设置为255
                    count += 1
        return count

    w, h = image.shape
    for _w in range(w):
        for _h in range(h):
            if _w == 0 or _h == 0:
                image[_w, _h] = 255
                continue
            # 计算邻域pixel值小于255的个数
            pixel = image[_w, _h]
            if pixel == 255:
                continue

            if calculate_noise_count(image, _w, _h) < k:
                image[_w, _h] = 255
    return image


def count_number(num_list, num):
    """
    统计一维数组中某个数字的个数
    :param num_list:
    :param num:
    :return: num的数量
    """
    t = 0
    for i in num_list:
        if i == num:
            t += 1
    return t


# 切割图片
def cut_vertical(img_list, c_value=255):
    """
    投影法竖直切割图片的数组
    :param img_list: 传入的数据为一个由（二维）图片构成的数组，不是单纯的图片
    :param c_value: 切割的值 c_value
    :return: 切割之后的图片的数组
    """
    # 如果传入的是一个普通的二值化的图片，则需要首先将这个二值化的图片升维为图片的数组
    if len(np.array(img_list).shape) == 2:
        img_list = img_list[None]
    r_list = []
    for img_i in img_list:
        end = 0
        for i in range(len(img_i.T)):
            if count_number(img_i.T[i], c_value) >= img_i.shape[0]:
                star = end
                end = i
                if end - star > 1:
                    r_list.append(img_i[:, star:end])
    return r_list
