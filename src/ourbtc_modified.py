# -*- coding: utf-8 -*-
"""Group Project
Authors: M. Lukasik, L. Smalec, R. Piatek
"""
import cv2 as cv
from skimage.metrics import structural_similarity
import numpy as np
from math import sqrt, log10
from bitarray import bitarray
from bitarray.util import ba2int
from pathlib import Path
import glob


def btc_encode(image, method, block_size):
    """BTC encoding with chosen method
        :param image: image in gray-scale (ndarray)
        :param method: 'standard', 'modified', 'ambtc', 'ours' (string)
        :param block_size: int

    Returnd:
        image_btc (bitarray)
    """
    if method == '8_1_0':
        image_btc = btc_ours_encode_8_1_0(image, block_size)
        return image_btc
    if method == '8_0_1':
        image_btc = btc_ours_encode_8_0_1(image, block_size)
        return image_btc


def btc_decode(image_btc, method, block_size):
    """BTC encoding with chosen method
        :param image_btc: bitarray
        :param method: 'standard', 'modified', 'ambtc', 'ours' (string)
        :param block_size: int

    Returnd:
        image_btc (ndarray)
    """
    if method == '8_1_0':
        image_btc = btc_ours_decode_1_0(image_btc, block_size)
        return image_btc
    if method == '8_0_1':
        image_btc = btc_ours_decode_0_1(image_btc, block_size)
        return image_btc


def make_tests0_1(block, xh, xl, block_size, block_checker, block_quant):

    test0 = np.zeros((4, 4))


    for i in range(0, block_size):
        for j in range(0, block_size):
            if block_checker[i, j] != 0:
                if block_quant[i, j] == 1:
                    test0[i, j] = xh
                else:
                    test0[i, j] = xl
            else:
                test0[i, j] = 256

    for i in range(0, block_size):
        for j in range(0, block_size):
            if i % 2 == 0 and j % 2 == 0:
                if i == 0 and j == 0:
                    test0[i, j] = (int(test0[i + 1, j]) + int(test0[i, j + 1])) // 2
                elif i == 0 and j > 0:
                    test0[i, j] = (int(test0[i, j - 1]) + int(test0[i, j + 1]) + int(test0[i + 1, j])) // 3
                elif i > 0 and j == 0:
                    test0[i, j] = (int(test0[i + 1, j]) + int(test0[i - 1, j]) + int(test0[i, j + 1])) // 3
                else:
                    test0[i, j] = (int(test0[i + 1, j]) + int(test0[i - 1, j]) +
                                  int(test0[i, j + 1]) + int(test0[i, j - 1])) // 4
            elif i % 2 == 1 and j % 2 == 1:
                if i == 3 and j == 3:
                    test0[i, j] = (int(test0[i - 1, j]) + int(test0[i, j - 1])) // 2
                elif i == 3 and j < 3:
                    test0[i, j] = (int(test0[i, j - 1]) + int(test0[i, j + 1]) + int(test0[i - 1, j])) // 3
                elif i < 3 and j == 3:
                    test0[i, j] = (int(test0[i - 1, j]) + int(test0[i + 1, j]) + int(test0[i, j - 1])) // 3
                elif i < 3 and j < 3:
                    test0[i, j] = (int(test0[i + 1, j]) + int(test0[i - 1, j]) +
                                  int(test0[i, j + 1]) + int(test0[i, j - 1])) // 4
    mse0 = np.mean((block - test0) ** 2)
    iter_test = 0
    while True:
        iter_test += 1
        # break
        test1 = np.zeros((4, 4))
        test2 = np.zeros((4, 4))
        test3 = np.zeros((4, 4))
        test4 = np.zeros((4, 4))

        for i in range(0, block_size):
            for j in range(0, block_size):
                if block_checker[i, j] == 1:
                    if block_quant[i, j] == 1:
                        test1[i, j] = xh + 1
                        if test1[i, j] > 255:
                            test1[i, j] = 255
                        test2[i, j] = xh - 1
                        if test2[i, j] < 0:
                            test2[i, j] = 0
                        test3[i, j] = xh
                        test4[i, j] = xh
                    else:
                        test1[i, j] = xl
                        test2[i, j] = xl
                        test3[i, j] = xl + 1
                        if test3[i, j] > 255:
                            test3[i, j] = 255
                        test4[i, j] = xl - 1
                        if test4[i, j] < 0:
                            test4[i, j] = 0
                else:
                    test1[i, j] = 256
                    test2[i, j] = 256
                    test3[i, j] = 256
                    test4[i, j] = 256

        tests = [test1, test2, test3, test4]
        # print(1, tests)
        for item in tests:
            for i in range(0, block_size):
                for j in range(0, block_size):
                    # if i % 2 == 1 and j % 2 == 0:
                    #     if i == 3 and j == 0:
                    #         item[i, j] = (int(item[i - 1, j]) + int(item[i, j + 1])) // 2
                    #     elif i < 3 and j == 0:
                    #         item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j])
                    #                                + int(item[i, j + 1])) // 3
                    #     elif i == 3 and 0 < j < 3:
                    #         item[i, j] = (int(item[i - 1, j]) + int(item[i, j + 1])
                    #                                + int(item[i, j - 1])) // 3
                    #     elif i < 3 and 0 < j < 3:
                    #         item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j])
                    #                                + int(item[i, j + 1]) + int(item[i, j - 1])) // 4
                    # elif i % 2 == 0 and j % 2 == 1:
                    #     # print(image_decoded[i, j])
                    #     if i == 0 and j == 3:
                    #         item[i, j] = (int(item[i + 1, j]) + int(item[i, j - 1])) // 2
                    #     elif i == 0 and j < 3:
                    #         item[i, j] = (int(item[i + 1, j]) + int(item[i, j + 1])
                    #                                + int(item[i, j - 1])) // 3
                    #     elif 0 < i < 3 and j == 3:
                    #         item[i, j] = (int(item[i - 1, j]) + int(item[i + 1, j])
                    #                                + int(item[i, j - 1])) // 3
                    #     elif 0 < i < 3 and j < 3:
                    #         item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j])
                    #                                + int(item[i, j + 1]) + int(item[i, j - 1])) // 4
                    if i % 2 == 0 and j % 2 == 0:
                        if i == 0 and j == 0:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i, j + 1])) // 2
                        elif i == 0 and j > 0:
                            item[i, j] = (int(item[i, j - 1]) + int(item[i, j + 1]) + int(item[i + 1, j])) // 3
                        elif i > 0 and j == 0:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j]) + int(item[i, j + 1])) // 3
                        else:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j]) +
                                          int(item[i, j + 1]) + int(item[i, j - 1])) // 4
                    elif i % 2 == 1 and j % 2 == 1:
                        if i == 3 and j == 3:
                            item[i, j] = (int(item[i - 1, j]) + int(item[i, j - 1])) // 2
                        elif i == 3 and j < 3:
                            item[i, j] = (int(item[i, j - 1]) + int(item[i, j + 1]) + int(item[i - 1, j])) // 3
                        elif i < 3 and j == 3:
                            item[i, j] = (int(item[i - 1, j]) + int(item[i + 1, j]) + int(item[i, j - 1])) // 3
                        elif i < 3 and j < 3:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j]) +
                                          int(item[i, j + 1]) + int(item[i, j - 1])) // 4
        mse1 = np.mean((block - tests[0]) ** 2)
        mse2 = np.mean((block - tests[1]) ** 2)
        mse3 = np.mean((block - tests[2]) ** 2)
        mse4 = np.mean((block - tests[3]) ** 2)
        mses = [mse1, mse2, mse3, mse4]
        if mse0 > min(mses):
            # print(2)
            mse0 = min(mses)
            mse_index = mses.index(mse0)
            if mse_index == 0:
                xh += 1
            if mse_index == 1:
                xh -= 1
            if mse_index == 2:
                xl += 1
            if mse_index == 3:
                xl -= 1
        elif mse0 <= min(mses):
            # mse_index = mses.index(min(mses))
            # if mse_index == 0:
            #     xh += 1
            # if mse_index == 1:
            #     xh -= 1
            # if mse_index == 2:
            #     xl += 1
            # if mse_index == 3:
            #     xl -= 1
            break
    if xh > 255:
        xh = 255
    if xl > 255:
        xl = 255
    if xh < 0:
        xh = 0
    if xl < 0:
        xl = 0
    return int(xh), int(xl), iter_test


def make_tests1_0(block, xh, xl, block_size, block_checker, block_quant):

    test0 = np.zeros((4, 4))


    for i in range(0, block_size):
        for j in range(0, block_size):
            if block_checker[i, j] != 0:
                if block_quant[i, j] == 1:
                    test0[i, j] = xh
                else:
                    test0[i, j] = xl
            else:
                test0[i, j] = 256

    for i in range(0, block_size):
        for j in range(0, block_size):
            if i % 2 == 0 and j % 2 == 0:
                if i == 0 and j == 0:
                    test0[i, j] = (int(test0[i + 1, j]) + int(test0[i, j + 1])) // 2
                elif i == 0 and j > 0:
                    test0[i, j] = (int(test0[i, j - 1]) + int(test0[i, j + 1]) + int(test0[i + 1, j])) // 3
                elif i > 0 and j == 0:
                    test0[i, j] = (int(test0[i + 1, j]) + int(test0[i - 1, j]) + int(test0[i, j + 1])) // 3
                else:
                    test0[i, j] = (int(test0[i + 1, j]) + int(test0[i - 1, j]) +
                                  int(test0[i, j + 1]) + int(test0[i, j - 1])) // 4
            elif i % 2 == 1 and j % 2 == 1:
                if i == 3 and j == 3:
                    test0[i, j] = (int(test0[i - 1, j]) + int(test0[i, j - 1])) // 2
                elif i == 3 and j < 3:
                    test0[i, j] = (int(test0[i, j - 1]) + int(test0[i, j + 1]) + int(test0[i - 1, j])) // 3
                elif i < 3 and j == 3:
                    test0[i, j] = (int(test0[i - 1, j]) + int(test0[i + 1, j]) + int(test0[i, j - 1])) // 3
                elif i < 3 and j < 3:
                    test0[i, j] = (int(test0[i + 1, j]) + int(test0[i - 1, j]) +
                                  int(test0[i, j + 1]) + int(test0[i, j - 1])) // 4
    mse0 = np.mean((block - test0) ** 2)

    iter_test = 0
    while True:
        iter_test += 1
        # break
        test1 = np.zeros((4, 4))
        test2 = np.zeros((4, 4))
        test3 = np.zeros((4, 4))
        test4 = np.zeros((4, 4))

        for i in range(0, block_size):
            for j in range(0, block_size):
                if block_checker[i, j] == 0:
                    if block_quant[i, j] == 1:
                        test1[i, j] = xh + 1
                        if test1[i, j] > 255:
                            test1[i, j] = 255
                        test2[i, j] = xh - 1
                        if test2[i, j] < 0:
                            test2[i, j] = 0
                        test3[i, j] = xh
                        test4[i, j] = xh
                    else:
                        test1[i, j] = xl
                        test2[i, j] = xl
                        test3[i, j] = xl + 1
                        if test3[i, j] > 255:
                            test3[i, j] = 255
                        test4[i, j] = xl - 1
                        if test4[i, j] < 0:
                            test4[i, j] = 0
                else:
                    test1[i, j] = 256
                    test2[i, j] = 256
                    test3[i, j] = 256
                    test4[i, j] = 256

        tests = [test1, test2, test3, test4]
        # print(1, tests)
        for item in tests:
            for i in range(0, block_size):
                for j in range(0, block_size):
                    if i % 2 == 1 and j % 2 == 0:
                        if i == 3 and j == 0:
                            item[i, j] = (int(item[i - 1, j]) + int(item[i, j + 1])) // 2
                        elif i < 3 and j == 0:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j])
                                                   + int(item[i, j + 1])) // 3
                        elif i == 3 and 0 < j < 3:
                            item[i, j] = (int(item[i - 1, j]) + int(item[i, j + 1])
                                                   + int(item[i, j - 1])) // 3
                        elif i < 3 and 0 < j < 3:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j])
                                                   + int(item[i, j + 1]) + int(item[i, j - 1])) // 4
                    elif i % 2 == 0 and j % 2 == 1:
                        # print(image_decoded[i, j])
                        if i == 0 and j == 3:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i, j - 1])) // 2
                        elif i == 0 and j < 3:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i, j + 1])
                                                   + int(item[i, j - 1])) // 3
                        elif 0 < i < 3 and j == 3:
                            item[i, j] = (int(item[i - 1, j]) + int(item[i + 1, j])
                                                   + int(item[i, j - 1])) // 3
                        elif 0 < i < 3 and j < 3:
                            item[i, j] = (int(item[i + 1, j]) + int(item[i - 1, j])
                                                   + int(item[i, j + 1]) + int(item[i, j - 1])) // 4
        mse1 = np.mean((block - tests[0]) ** 2)
        mse2 = np.mean((block - tests[1]) ** 2)
        mse3 = np.mean((block - tests[2]) ** 2)
        mse4 = np.mean((block - tests[3]) ** 2)
        mses = [mse1, mse2, mse3, mse4]
        if mse0 > min(mses):
            # print(2)
            mse0 = min(mses)
            mse_index = mses.index(mse0)
            if mse_index == 0:
                xh += 1
            if mse_index == 1:
                xh -= 1
            if mse_index == 2:
                xl += 1
            if mse_index == 3:
                xl -= 1
        elif mse0 <= min(mses):
            # mse_index = mses.index(min(mses))
            # if mse_index == 0:
            #     xh += 1
            # if mse_index == 1:
            #     xh -= 1
            # if mse_index == 2:
            #     xl += 1
            # if mse_index == 3:
            #     xl -= 1
            break
    if xh > 255:
        xh = 255
    if xl > 255:
        xl = 255
    if xh < 0:
        xh = 0
    if xl < 0:
        xl = 0
    return int(xh), int(xl), iter_test


def btc_ours_encode_8_0_1(image_gray, block_size):
    """Compresses image file using modified BTC
        :param image_gray: ndarray
        :param block_size: int

    Returns:
        image_btc (binary)
    """
    image_btc = bitarray()
    param_btc = bitarray()
    size = bitarray()
    size.extend("{0:016b}".format(image_gray.shape[0]))
    size.extend("{0:016b}".format(image_gray.shape[1]))
    iters = []
    for w in range(0, image_gray.shape[0], block_size):
        for h in range(0, image_gray.shape[1], block_size):
            block = image_gray[w: w + block_size, h: h + block_size]
            block_quant = np.array(block)
            block_quant_new = np.zeros((block_size, block_size))

            block_checker = np.zeros((4, 4), dtype=int)
            block_checker[1::2, ::2] = 1
            block_checker[::2, 1::2] = 1

            for_mean = []
            block_8 = []
            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block_checker[i, j] == 1:
                        for_mean.append(block[i, j])
            mean = np.mean(for_mean)

            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block[i, j] >= mean:
                        block_quant[i, j] = 1
                    else:
                        block_quant[i, j] = 0
                    if block_checker[i, j] == 1:
                        block_8.append(block_quant[i, j])

            for a in range(0, 10):
                unq, ids, count = np.unique(block_8, return_inverse=True, return_counts=True)
                out = np.column_stack((unq, np.bincount(ids, for_mean) / count))

                block_8 = []
                for i in range(0, block_size):
                    for j in range(0, block_size):
                        if out.shape[0] == 2:
                            if min([out[0, 1], out[1, 1]], key=lambda x: abs(x - block[i, j])) == out[0, 1]:
                                block_quant[i, j] = 0
                            else:
                                block_quant[i, j] = 1
                        else:
                            block_quant[i, j] = 0
                        if block_checker[i, j] == 1:
                            block_8.append(block_quant[i, j])
                if np.array_equal(block_quant_new, block_quant):
                    break
                else:
                    block_quant_new = np.array(block_quant)

            xl = out[0, 1]
            if out.shape[0] == 2:
                xh = out[1, 1]
            else:
                xh = out[0, 1]
            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block_checker[i, j] != 0:
                        if block_quant[i, j] == 1:
                            image_btc.append(True)
                        else:
                            image_btc.append(False)

            xh1, xl1, iter_test = make_tests0_1(block, xh, xl, block_size, block_checker, block_quant)
            # if out.shape[0] == 2:
            #     param_btc.extend("{0:08b}".format(int(out[0, 1])))
            #     param_btc.extend("{0:08b}".format(int(out[1, 1])))
            # else:
            #     param_btc.extend("{0:08b}".format(int(out[0, 1])))
            #     param_btc.extend("{0:08b}".format(int(out[0, 1])))
            param_btc.extend("{0:08b}".format(int(xl1)))
            param_btc.extend("{0:08b}".format(int(xh1)))
            iters.append(iter_test)
    image_btc.extend(param_btc)
    image_btc.extend(size)
    return image_btc, iters


def btc_ours_encode_8_1_0(image_gray, block_size):
    """Compresses image file using modified BTC
        :param image_gray: ndarray
        :param block_size: int

    Returns:
        image_btc (binary)
    """
    image_btc = bitarray()
    param_btc = bitarray()
    size = bitarray()
    size.extend("{0:016b}".format(image_gray.shape[0]))
    size.extend("{0:016b}".format(image_gray.shape[1]))
    iters = []
    for w in range(0, image_gray.shape[0], block_size):
        for h in range(0, image_gray.shape[1], block_size):
            block = image_gray[w: w + block_size, h: h + block_size]
            block_quant = np.array(block)
            block_quant_new = np.zeros((block_size, block_size))

            block_checker = np.zeros((4, 4), dtype=int)
            block_checker[1::2, ::2] = 1
            block_checker[::2, 1::2] = 1

            for_mean = []
            block_8 = []
            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block_checker[i, j] == 0:
                        for_mean.append(block[i, j])
            mean = np.mean(for_mean)

            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block[i, j] >= mean:
                        block_quant[i, j] = 1
                    else:
                        block_quant[i, j] = 0
                    if block_checker[i, j] == 0:
                        block_8.append(block_quant[i, j])

            for a in range(0, 10):
                unq, ids, count = np.unique(block_8, return_inverse=True, return_counts=True)
                out = np.column_stack((unq, np.bincount(ids, for_mean) / count))

                block_8 = []
                for i in range(0, block_size):
                    for j in range(0, block_size):
                        if out.shape[0] == 2:
                            if min([out[0, 1], out[1, 1]], key=lambda x: abs(x - block[i, j])) == out[0, 1]:
                                block_quant[i, j] = 0
                            else:
                                block_quant[i, j] = 1
                        else:
                            block_quant[i, j] = 0
                        if block_checker[i, j] == 0:
                            block_8.append(block_quant[i, j])
                if np.array_equal(block_quant_new, block_quant):
                    break
                else:
                    block_quant_new = np.array(block_quant)

            xl = out[0, 1]
            if out.shape[0] == 2:
                xh = out[1, 1]
            else:
                xh = out[0, 1]


            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block_checker[i, j] != 1:
                        if block_quant[i, j] == 1:
                            image_btc.append(True)
                        else:
                            image_btc.append(False)

            xh1, xl1, iter_test = make_tests1_0(block, xh, xl, block_size, block_checker, block_quant)

            param_btc.extend("{0:08b}".format(int(xl1)))
            param_btc.extend("{0:08b}".format(int(xh1)))
            iters.append(iter_test)
    image_btc.extend(param_btc)
    image_btc.extend(size)
    return image_btc, iters


def btc_ours_decode_0_1(image_btc, block_size):
    """Decodes bool array using BTC
        :param image_btc: binary file
        :param block_size: int

    Returns:
        image (ndarray)
    """
    width = ba2int(image_btc[-32:-16])
    height = ba2int(image_btc[-16:])
    image_decoded = np.zeros((width, height), dtype="uint8")
    n1 = 0
    n2 = 8
    for w in range(0, height * width // 2, height * block_size // 2):
        for b in range(0, height * block_size // 2, block_size ** 2 // 2):
            block = np.zeros((block_size, block_size), dtype=int)
            block_checker = np.zeros((4, 4), dtype=int)
            block_checker[1::2, ::2] = 1
            block_checker[::2, 1::2] = 1
            bit_list = image_btc[w + b: w + b + (block_size ** 2 // 2)].tolist()
            for i in range(block_size):
                for j in range(block_size):
                    if block_checker[i, j] == 1:
                        if bit_list[0]:
                            block[i, j] = 1
                        else:
                            block[i, j] = 0
                        bit_list.pop(0)
                    else:
                        block[i, j] = 256
                        # bit_list.pop(0)

            block = np.where(block == 1,
                             ba2int(image_btc[height * width // 2 + n2: height * width // 2 + n2 + 8]),
                             np.where(block == 0,
                                      ba2int(image_btc[height * width // 2 + n1: height * width // 2 + n2]),
                                      256))

            image_decoded[int(w * 2 / height): int(w * 2 / height) + block_size,
                          int(b * 2 / block_size): int(b * 2 / block_size) + block_size] = block
            n1 += 16
            n2 += 16
    for i in range(width):
        for j in range(height):
            if i % 2 == 0 and j % 2 == 0:
                if i == 0 and j == 0:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i, j + 1])) // 2
                elif i == 0 and j > 0:
                    image_decoded[i, j] = (int(image_decoded[i, j - 1]) + int(image_decoded[i, j + 1])
                                           + int(image_decoded[i + 1, j])) // 3
                elif i > 0 and j == 0:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i - 1, j])
                                           + int(image_decoded[i, j + 1])) // 3
                else:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i - 1, j])
                                           + int(image_decoded[i, j + 1]) + int(image_decoded[i, j - 1])) // 4
            elif i % 2 == 1 and j % 2 == 1:
                if i == width - 1 and j == height - 1:
                    image_decoded[i, j] = (int(image_decoded[i - 1, j]) + int(image_decoded[i, j - 1])) // 2
                elif i == width - 1 and j < height - 1:
                    image_decoded[i, j] = (int(image_decoded[i, j - 1]) + int(image_decoded[i, j + 1])
                                           + int(image_decoded[i - 1, j])) // 3
                elif i < width - 1 and j == height - 1:
                    image_decoded[i, j] = (int(image_decoded[i - 1, j]) + int(image_decoded[i + 1, j])
                                           + int(image_decoded[i, j - 1])) // 3
                elif i < width - 1 and j < height - 1:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i - 1, j])
                                           + int(image_decoded[i, j + 1]) + int(image_decoded[i, j - 1])) // 4
    return image_decoded


def btc_ours_decode_1_0(image_btc, block_size):
    """Decodes bool array using BTC
        :param image_btc: binary file
        :param block_size: int

    Returns:
        image (ndarray)
    """
    width = ba2int(image_btc[-32:-16])
    height = ba2int(image_btc[-16:])
    image_decoded = np.zeros((width, height), dtype="uint8")
    n1 = 0
    n2 = 8
    for w in range(0, height * width // 2, height * block_size // 2):
        for b in range(0, height * block_size // 2, block_size ** 2 // 2):
            block = np.zeros((block_size, block_size), dtype=int)
            block_checker = np.zeros((4, 4), dtype=int)
            block_checker[1::2, ::2] = 1
            block_checker[::2, 1::2] = 1
            bit_list = image_btc[w + b: w + b + (block_size ** 2 // 2)].tolist()
            for i in range(block_size):
                for j in range(block_size):
                    if block_checker[i, j] == 0:
                        if bit_list[0]:
                            block[i, j] = 1
                        else:
                            block[i, j] = 0
                        bit_list.pop(0)
                    else:
                        block[i, j] = 256
                        # bit_list.pop(0)

            block = np.where(block == 1,
                             ba2int(image_btc[height * width // 2 + n2: height * width // 2 + n2 + 8]),
                             np.where(block == 0,
                                      ba2int(image_btc[height * width // 2 + n1: height * width // 2 + n2]),
                                      256))

            image_decoded[int(w * 2 / height): int(w * 2 / height) + block_size,
                          int(b * 2 / block_size): int(b * 2 / block_size) + block_size] = block
            n1 += 16
            n2 += 16
    for i in range(width):
        for j in range(height):
            if i % 2 == 1 and j % 2 == 0:
                if i == width - 1 and j == 0:
                    image_decoded[i, j] = (int(image_decoded[i - 1, j]) + int(image_decoded[i, j + 1])) // 2
                elif i < width - 1 and j == 0:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i - 1, j])
                                           + int(image_decoded[i, j + 1])) // 3
                elif i == width - 1 and 0 < j < height - 1:
                    image_decoded[i, j] = (int(image_decoded[i - 1, j]) + int(image_decoded[i, j + 1])
                                           + int(image_decoded[i, j - 1])) // 3
                elif i < width - 1 and 0 < j < height - 1:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i - 1, j])
                                           + int(image_decoded[i, j + 1]) + int(image_decoded[i, j - 1])) // 4
            elif i % 2 == 0 and j % 2 == 1:
                # print(image_decoded[i, j])
                if i == 0 and j == height - 1:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i, j - 1])) // 2
                elif i == 0 and j < height - 1:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i, j + 1])
                                           + int(image_decoded[i, j - 1])) // 3
                elif 0 < i < width - 1 and j == height - 1:
                    image_decoded[i, j] = (int(image_decoded[i - 1, j]) + int(image_decoded[i + 1, j])
                                           + int(image_decoded[i, j - 1])) // 3
                elif 0 < i < width - 1 and j < height - 1:
                    image_decoded[i, j] = (int(image_decoded[i + 1, j]) + int(image_decoded[i - 1, j])
                                           + int(image_decoded[i, j + 1]) + int(image_decoded[i, j - 1])) // 4

    return image_decoded


def psnr(source, processed):
    """Quality assessment Peak Signal-to-Noise Ratio
        :param source: source image in gray-scale
        :param processed: processed image in gray-scale

    Returns:
        quality_index: in dB
    """
    mse = np.mean((source - processed) ** 2)  # mean squared error
    if mse == 0:
        return 100
    max_pixel = 255.0
    quality_index = 20 * log10(max_pixel / sqrt(mse))
    return quality_index


def ssim(source, processed):
    """Quality assessment Structural Similarity Index
        :param source: source image in gray-scale
        :param processed: processed image in gray-scale

    Returns:
        score: in dB
    """
    (score, diff) = structural_similarity(source, processed, full=True)
    return score


def raport(image, im_path, images_decompressed, method):
    print(im_path)
    cr_value = (Path(im_path).stat().st_size-1078+32) / Path(method+".bin").stat().st_size
    print(f'CR value for {method} is: '
          f'{cr_value}')
    quality_value = psnr(image, images_decompressed)
    print(f"PSNR value for {method} is {quality_value} dB")
    ssim_value = ssim(image, images_decompressed)
    print(f"SSIM value for {method} is {ssim_value}")
    print()

    return cr_value, quality_value, ssim_value


def to_file(im_path, raports, txt_file):
    txt_file.write(f'{im_path[15:-4]}'
                   f' & {format(round(raports[0][1], 2), ".2f")} & {format(round(raports[0][2], 4), ".4f")}'
                   f' & {format(round(raports[1][1], 2), ".2f")} & {format(round(raports[1][2], 4), ".4f")} \\\\\n'
                   f'\hline\n')


def main():
    images = glob.glob("Testowe_obrazy/*.bmp")
    print(images)
    block_size = 4
    for image_path in images:
        im = cv.imread(image_path)
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        images = []
        methods = ['8_0_1', '8_1_0']

        badania = open('modified_our.txt', 'a')
        # iters = []
        for method in methods:
            image_compressed, iter_test = btc_encode(im_gray, method, block_size)
            with open(method+'.bin', 'wb') as fh:
                image_compressed.tofile(fh)
            # iters.append(iter_test)
            images.append(btc_decode(image_compressed, method, block_size))

        # badania.write(f'{np.unique(iters, return_counts=True)[0]} {np.unique(iters, return_counts=True)[1]}\n')
        # cv.imshow('original', im)
        raports = []
        for i, image in enumerate(images):
            raports.append(raport(im_gray, image_path, image, methods[i]))
            # cv.imshow(methods[i], image)
            # cv.imwrite(methods[i]+'.png', image)

        to_file(image_path, raports, badania)
        # cv.waitKey()


        # image_compressed = btc_encode(im_gray, 'standard', block_size)
        # temp = btc_decode(image_compressed, 'standard', block_size)
        # cv.imshow('tmp', temp)
        # raport(im_gray, "Testowe_obrazy/crowd512.bmp", temp, 'modified')


if __name__ == "__main__":
    main()
