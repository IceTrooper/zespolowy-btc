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
    if method == '16_1_0':
        image_btc = btc_ours_encode_16_1_0(image, block_size)
        return image_btc
    if method == '16_0_1':
        image_btc = btc_ours_encode_16_0_1(image, block_size)
        return image_btc
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
    if method == '16_1_0':
        image_btc = btc_ours_decode_1_0(image_btc, block_size)
        return image_btc
    if method == '16_0_1':
        image_btc = btc_ours_decode_0_1(image_btc, block_size)
        return image_btc
    if method == '8_1_0':
        image_btc = btc_ours_decode_1_0(image_btc, block_size)
        return image_btc
    if method == '8_0_1':
        image_btc = btc_ours_decode_0_1(image_btc, block_size)
        return image_btc


def btc_ours_encode_16_0_1(image_gray, block_size):
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
    for w in range(0, image_gray.shape[0], block_size):
        for h in range(0, image_gray.shape[1], block_size):
            block = image_gray[w: w + block_size, h: h + block_size]
            block_quant = np.array(block)
            mean = np.mean(block)
            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block[i, j] >= mean:
                        block_quant[i, j] = 1
                    else:
                        block_quant[i, j] = 0

            for a in range(0, 10):
                unq, ids, count = np.unique(np.reshape(block_quant, block_size**2), return_inverse=True, return_counts=True)
                out = np.column_stack((unq, np.bincount(ids, np.reshape(block, block_size**2))/count))
                for i in range(0, block_size):
                    for j in range(0, block_size):
                        if out.shape[0] == 2:
                            if min([out[0, 1], out[1, 1]], key=lambda x: abs(x-block[i, j])) == out[0, 1]:
                                block_quant[i, j] = 0
                            else:
                                block_quant[i, j] = 1
                        else:
                            block_quant[i, j] = 0

            block_checker = np.zeros((4, 4), dtype=int)
            block_checker[1::2, ::2] = 1
            block_checker[::2, 1::2] = 1
            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block_checker[i, j] != 0:
                        if block_quant[i, j] == 1:
                            image_btc.append(True)
                        else:
                            image_btc.append(False)

            if out.shape[0] == 2:
                param_btc.extend("{0:08b}".format(int(out[0, 1])))
                param_btc.extend("{0:08b}".format(int(out[1, 1])))
            else:
                param_btc.extend("{0:08b}".format(int(out[0, 1])))
                param_btc.extend("{0:08b}".format(int(out[0, 1])))

    image_btc.extend(param_btc)
    image_btc.extend(size)
    return image_btc


def btc_ours_encode_16_1_0(image_gray, block_size):
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
    for w in range(0, image_gray.shape[0], block_size):
        for h in range(0, image_gray.shape[1], block_size):
            block = image_gray[w: w + block_size, h: h + block_size]
            block_quant = np.array(block)
            mean = np.mean(block)
            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block[i, j] >= mean:
                        block_quant[i, j] = 1
                    else:
                        block_quant[i, j] = 0

            for a in range(0, 10):
                unq, ids, count = np.unique(np.reshape(block_quant, block_size**2), return_inverse=True, return_counts=True)
                out = np.column_stack((unq, np.bincount(ids, np.reshape(block, block_size**2))/count))
                for i in range(0, block_size):
                    for j in range(0, block_size):
                        if out.shape[0] == 2:
                            if min([out[0, 1], out[1, 1]], key=lambda x: abs(x-block[i, j])) == out[0, 1]:
                                block_quant[i, j] = 0
                            else:
                                block_quant[i, j] = 1
                        else:
                            block_quant[i, j] = 0

            block_checker = np.zeros((4, 4), dtype=int)
            block_checker[1::2, ::2] = 1
            block_checker[::2, 1::2] = 1
            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block_checker[i, j] != 1:
                        if block_quant[i, j] == 1:
                            image_btc.append(True)
                        else:
                            image_btc.append(False)

            if out.shape[0] == 2:
                param_btc.extend("{0:08b}".format(int(out[0, 1])))
                param_btc.extend("{0:08b}".format(int(out[1, 1])))
            else:
                param_btc.extend("{0:08b}".format(int(out[0, 1])))
                param_btc.extend("{0:08b}".format(int(out[0, 1])))

    image_btc.extend(param_btc)
    image_btc.extend(size)
    return image_btc


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
                    if block_checker[i,j] == 1:
                        block_8.append(block_quant[i, j])

            for a in range(0, 10):
                unq, ids, count = np.unique(block_8, return_inverse=True, return_counts=True)
                out = np.column_stack((unq, np.bincount(ids, for_mean)/count))

                block_8 = []
                for i in range(0, block_size):
                    for j in range(0, block_size):
                        if out.shape[0] == 2:
                            if min([out[0, 1], out[1, 1]], key=lambda x: abs(x-block[i, j])) == out[0, 1]:
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

            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block_checker[i, j] != 0:
                        if block_quant[i, j] == 1:
                            image_btc.append(True)
                        else:
                            image_btc.append(False)

            if out.shape[0] == 2:
                param_btc.extend("{0:08b}".format(int(out[0, 1])))
                param_btc.extend("{0:08b}".format(int(out[1, 1])))
            else:
                param_btc.extend("{0:08b}".format(int(out[0, 1])))
                param_btc.extend("{0:08b}".format(int(out[0, 1])))

    image_btc.extend(param_btc)
    image_btc.extend(size)
    return image_btc


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

            for i in range(0, block_size):
                for j in range(0, block_size):
                    if block_checker[i, j] != 1:
                        if block_quant[i, j] == 1:
                            image_btc.append(True)
                        else:
                            image_btc.append(False)

            if out.shape[0] == 2:
                param_btc.extend("{0:08b}".format(int(out[0, 1])))
                param_btc.extend("{0:08b}".format(int(out[1, 1])))
            else:
                param_btc.extend("{0:08b}".format(int(out[0, 1])))
                param_btc.extend("{0:08b}".format(int(out[0, 1])))

    image_btc.extend(param_btc)
    image_btc.extend(size)
    return image_btc


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
                   f' & {format(round(raports[1][1], 2), ".2f")} & {format(round(raports[1][2], 4), ".4f")}'
                   f' & {format(round(raports[2][1], 2), ".2f")} & {format(round(raports[2][2], 4), ".4f")}'
                   f' & {format(round(raports[3][1], 2), ".2f")} & {format(round(raports[3][2], 4), ".4f")} \\\\\n'
                   f'\hline\n')


def main():
    images = glob.glob("Testowe_obrazy/*.bmp")
    print(images)
    block_size = 4
    for image_path in images:
        im = cv.imread(image_path)
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        images = []
        methods = ['16_0_1', '16_1_0', '8_0_1', '8_1_0']

        badania = open('badania_ourbtc.txt', 'a')

        for method in methods:
            image_compressed = btc_encode(im_gray, method, block_size)
            with open(method+'.bin', 'wb') as fh:
                image_compressed.tofile(fh)

            images.append(btc_decode(image_compressed, method, block_size))

        # cv.imshow('original', im)
        raports = []
        for i, image in enumerate(images):
            raports.append(raport(im_gray, image_path, image, methods[i]))
            # cv.imshow(methods[i], image)
            # cv.imwrite(methods[i]+'.png', image)

        to_file(image_path, raports, badania)
        # cv.waitKey()


if __name__ == "__main__":
    main()
