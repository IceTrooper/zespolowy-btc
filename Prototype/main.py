# -*- coding: utf-8 -*-
"""Group Project
Authors: M. Lukasik, L. Smalec, R. Piatek
"""
import cv2 as cv
import numpy as np
from math import sqrt, log10


def btc_encode(image_gray, block_size):
    """Compresses image file using BTC
        :param image_gray: ndarray
        :param block_size: int

    Returns:
        image_btc (ndarray)
        btc_tuple (array)
    """
    image_btc = np.zeros((image_gray.shape[0], image_gray.shape[1]), dtype="bool")
    btc_tuple = []
    for w in range(0, image_gray.shape[0], block_size):
        for h in range(0, image_gray.shape[1], block_size):
            block = image_gray[w: w + block_size, h: h + block_size]
            mean = np.mean(block)
            std = np.std(block)
            q = np.sum(block > mean)
            a = mean - (std * sqrt(q / (block_size**2 - q)))
            b = a if q == 0 else mean + (std * sqrt((block_size**2 - q) / q))
            block = np.where(block >= mean, True, False)
            image_btc[w: w + block_size, h: h+block_size] = block
            btc_tuple.append((int(a), int(b)))
    return image_btc, btc_tuple


def btc_decode(image_btc, btc_tuple, block_size):
    """Decodes bool array using BTC
        :param image_btc: ndarray of boolean values
        :param btc_tuple: array with quantized values
        :param block_size: int

    Returns:
        image (ndarray)
    """
    image_decoded = np.zeros((image_btc.shape[0], image_btc.shape[1]), dtype="uint8")
    for w in range(0, image_btc.shape[0], block_size):
        for h in range(0, image_btc.shape[1], block_size):
            block = image_btc[w: w + block_size, h: h + block_size]
            block = np.where(block == True, btc_tuple[0][1], btc_tuple[0][0])
            btc_tuple.pop(0)
            image_decoded[w: w + block_size, h: h + block_size] = block
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


def main():
    im = cv.imread('TestImages/BOARD.bmp')
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_btc, btc_tuple = btc_encode(im_gray, 4)
    im_dec = btc_decode(im_btc, btc_tuple, 4)

    cv.imshow('before', im)
    cv.imshow('after', im_dec)
    cv.waitKey()
    cv.imwrite('test.png', im_dec)
    quality_value = psnr(im_gray, im_dec)
    print(f"PSNR value is {quality_value} dB")


if __name__ == "__main__":
    main()
