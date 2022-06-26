"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
EPS = 0.0001

import cv2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208925578


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    try:
        if representation != LOAD_GRAY_SCALE and representation != LOAD_RGB:
            raise Exception("Invalid input for representation")
        img_read = cv2.imread(filename=filename)
        # converting to the right scale
        img = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY) if representation == LOAD_GRAY_SCALE \
            else cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        # to float64 type
        img = img.astype(np.float64)
        # normalizing
        img_norm = np.zeros((img.shape[0], img.shape[1]))
        img_norm = cv2.normalize(img, img_norm, 0, 1, cv2.NORM_MINMAX)
        return img_norm
    except Exception as e:
        print("Exception has occurred ")
        print(e)


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    try:
        img = imReadAndConvert(filename=filename, representation=representation)
        if representation == LOAD_GRAY_SCALE:  # gray image
            plt.gray()
        plt.imshow(img)
        plt.show()
        return img
    except Exception as e:
        print("Exception has occurred")
        print(e)


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # Y = 0.299*R + 0.587*G + 0.114*B, I = 0.596*R – 0.275*G – 0.321*B, Q = 0.212*R – 0.523*G + 0.311*B
    try:
        yiq_values = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
        return np.dot(imgRGB, yiq_values.transpose())
    except TypeError as e:
        print("image is None")
        print(e)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # R = Y + 0.9469 * I + 0.6236 * Q, G = Y + 0.2748 * I - 0.6357 * Q, B = Y - 1.1 * I, 1.7 * Q
    try:
        rgb_values = np.array([[1, 0.9469, 0.6236], [1, -0.2748, -0.6357], [1, -1.1, 1.7]])
        return np.dot(imgYIQ, rgb_values.transpose())
    except TypeError as e:
        print("image is None")
        print(e)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    try:
        rgb = False
        if len(imgOrig.shape) == 3:  # if RGB, should take only Y channel
            rgb = True
            img_yiq = transformRGB2YIQ(imgOrig)
            imgOrig = img_yiq[:, :, 0]
        # normalizing to [0,255]
        imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
        imgOrig = imgOrig.astype('uint8')
        hist_orig = np.histogram(imgOrig.flatten(), bins=256)[0]  # output 2, orig hist
        orig_cumsum = np.cumsum(hist_orig)
        # new_img_cumsum = orig_cumsum[imgOrig]
        # Create a LookUpTable(LUT)
        lookup_table = np.floor((orig_cumsum / orig_cumsum.max()) * 255)
        # creating the new image by replacing each pixel with the corresponding pixel from the lu table
        new_img = np.zeros_like(imgOrig, dtype=float)
        for i in range(256):
            new_img[imgOrig == i] = int(lookup_table[i])
        new_hist = np.histogram(new_img.flatten(), bins=256)[0]  # output 3
        if rgb:
            img_yiq[:, :, 0] = new_img / 255  # normalizing back to [0,1]
            new_img = transformYIQ2RGB(img_yiq)  # output 1
        else:
            new_img = new_img / 255
        # plotting the orig image and the new image
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(imgOrig)
        ax[1].imshow(new_img)
        plt.show()
        return new_img, hist_orig, new_hist
    except TypeError as e:
        print("image is None")
        print(e)


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    try:
        rgb = False
        if imOrig.ndim == 3:
            rgb = True
            yiq_img = transformRGB2YIQ(imOrig)
            imOrig = yiq_img[:, :, 0]
        # make some empty lists for borders(z), images and errors
        q_img_list = []
        error_list = []
        # flatten image
        imOrig_f = (imOrig.flatten() * 255).astype(int)
        # histogram, did not work well with np.histogram
        img_hist = np.zeros(256)
        for n in imOrig_f:
            img_hist[n] += 1
        # init boundaries
        z = []
        border = img_hist.sum() / nQuant
        sum = 0
        z.append(0)
        for i in range(256):
            sum += img_hist[i]
            if sum >= border:
                sum = 0
                z.append(i + 1)
        z.append(256)
        # iterate nIter times
        for i in range(nIter):
            new_img = np.zeros(shape=imOrig.shape)
            # calculating the values of the segments intensities
            q = [int(np.average(range(z[i], z[i + 1]), weights=img_hist[z[i]: z[i + 1]])) for i in range(nQuant)]
            # calculating the new image with the new intensities
            for i in range(len(q)):
                new_img[imOrig > z[i] / 255] = q[i]
            # changing boundaries
            for i in range(1, len(z) - 1):
                z[i] = int((q[i - 1] + q[i]) / 2)
            # calculating mse and adding it to error list
            error_list.append(np.sqrt((imOrig * 255 - new_img) ** 2).mean())
            # adding to list
            new_img = new_img / 255
            q_img_list.append(new_img)
            # in case mse is converges
            if len(error_list) > 1:
                if error_list[-1] - error_list[-2] < EPS:
                    break
        if rgb:
            for img in range(len(q_img_list)):
                yiq_img[:, :, 0] = q_img_list[img]
                q_img_list[img] = transformYIQ2RGB(yiq_img)
        # plotting the errors list
        plt.plot(error_list)
        plt.show()
        return q_img_list, error_list
    except TypeError as e:
        print("one of argument is not in its correct type")
        print(e)
