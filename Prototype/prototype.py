import cv2 as cv
import numpy as np
import math

BLOCK_SIZE = 16

im = cv.imread('TestImages/lennagrey.bmp')
im_grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im_btc = np.zeros((im_grey.shape[0], im_grey.shape[1]), dtype="uint8")
block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype="uint8")
for w in range(0, im_grey.shape[0], BLOCK_SIZE):
    for h in range(0, im_grey.shape[1], BLOCK_SIZE):
        block = im_grey[w: w + BLOCK_SIZE, h: h + BLOCK_SIZE]
        mean = np.mean(block)
        std = np.std(block)
        q = np.sum(block > mean)
        a = mean - (std * math.sqrt(q / (BLOCK_SIZE**2 - q)))
        b = a if q == 0 else mean + (std * math.sqrt((BLOCK_SIZE**2 - q) / q))
        block = np.where(block >= mean, b, a)
        im_btc[w: w + BLOCK_SIZE, h: h+BLOCK_SIZE] = block

cv.imshow('before', im_grey)
cv.imshow('after', im_btc)
print(im_btc)
cv.waitKey()
cv.imwrite('test.png', im_btc)
