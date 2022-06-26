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
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keyboard
import ex1_utils
from ex1_utils import LOAD_GRAY_SCALE


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    try:
        global img
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if rep == LOAD_GRAY_SCALE else cv2.imread(img_path)
        cv2.namedWindow("Gamma Correction")
        trackbar = 'gamma %d' % 200
        cv2.createTrackbar(trackbar, "Gamma Correction", 0, 200, on_trackbar)
        on_trackbar(0)
        cv2.waitKey()
    except TypeError as e:
        print("img_path or/and rep are not in the correct type")
        print(e)

def on_trackbar(val):
    gamma = float(val / 100)
    inv_gamma = 1000 if gamma == 0 else 1.0 / gamma
    # calculating the new intensities according the gamma and mapping to the new image
    new_inten = np.array([((i / 255) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    new_img = cv2.LUT(img, new_inten)
    cv2.imshow("Gamma Correction", new_img)


def main():
    gammaDisplay('dark.jpg', 2)


if __name__ == '__main__':
    main()
