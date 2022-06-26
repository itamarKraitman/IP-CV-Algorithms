import math
import sys
from typing import List, Tuple

import numpy as np
import cv2
from numpy import ndarray
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from scipy import signal as sig

EPS = 0.0000001


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208925578


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    w = win_size // 2  # assuming win_size is odd
    k_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    k_y = k_x.T

    # calculating Ix,Iy,It for each pixel
    i_x = cv2.filter2D(im2, -1, k_x, borderType=cv2.BORDER_REPLICATE)
    i_y = cv2.filter2D(im2, -1, k_y, borderType=cv2.BORDER_REPLICATE)
    i_t = im2 - im1

    uv = []
    orig_points = []
    row, col = im1.shape[0], im1.shape[1]
    # for each pixel, solve the linear equation and calculate the eigen values
    for i in range(win_size, row - win_size + 1, step_size):
        for j in range(win_size, col - win_size + 1, step_size):
            new_x = i_x[i - w:i + w + 1, j - w:j + w + 1].flatten()
            new_y = i_y[i - w:i + w + 1, j - w:j + w + 1].flatten()
            new_t = i_t[i - w:i + w + 1, j - w:j + w + 1].flatten()

            A = np.array(
                [[sum(new_x[k] ** 2 for k in range(len(new_x))), sum(new_x[k] * new_y[k] for k in range(len(new_x)))],
                 [sum(new_x[k] * new_y[k] for k in range(len(new_x))), sum(new_y[k] ** 2 for k in range(len(new_y)))]])
            b = np.array([[-(sum(new_x[k] * new_t[k] for k in range(len(new_x)))),
                           -(sum(new_y[k] * new_t[k] for k in range(len(new_y))))]]).reshape(2, 1)

            try:
                ev1, ev2 = np.linalg.eigvals(A)
                if ev2 < ev1:
                    temp = ev1
                    ev1 = ev2
                    ev2 = temp
                # continue if the conditions are held
                if ev2 >= ev1 > 1 and ev2 / ev1 < 100:  # check the conditions
                    u_v = np.dot(np.linalg.pinv(A), b)
                    u = u_v[0][0]
                    v = u_v[1][0]
                    uv.append(np.array([u, v]))
                    orig_points.append(np.array([j, i]))
            except LinAlgError as e:
                print("error has occurred")
                print(e)
    return np.array(orig_points), np.array(uv)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    pyr_img1 = gaussianPyr(img1, k)
    pyr_img2 = gaussianPyr(img2, k)
    pyr_img1.reverse()
    pyr_img2.reverse()

    pts, uv = opticalFlow(pyr_img1[k - 1], pyr_img2[k - 1], stepSize, winSize)
    for i in range(k - 2, -1, -1):
        pts_i, uv_i = opticalFlow(pyr_img1[i], pyr_img2[i], stepSize, winSize)
        pts_i *= 2
        uv_i *= 2
        pts = np.vstack((pts, pts_i))
        uv = np.vstack((uv, uv_i))

    # creating result array
    u, v = [], []
    for i in range(uv.shape[0]):
        u.append(uv[i, 0])
        v.append(uv[i, 1])
    result = [[u[i] for i in range(len(u))],
              [v[j] for j in range(len(v))]]

    # plotting- for testing!
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img1, t, img1.shape[::-1])

    displayOpticalFlow(img_2, pts, uv)

    return np.asarray(result)


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("LK hierarchical")
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    # saving fig as image for comparing with Q1
    fig.savefig("lk_pyr.jpg", bbox_inches='tight')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pts, uv = opticalFlow(im1, im2, 20, 5)
    min_mse = np.inf
    trans_matrix = np.asarray([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]).astype(float)
    for i in range(uv.shape[0]):
        tx = uv[i, 0]
        ty = uv[i, 1]
        cand_trans_mat = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]]).astype(float)
        warp_img = cv2.warpPerspective(im1, cand_trans_mat, (im1.shape[1], im1.shape[0]))
        curr_mse = mse(im2, warp_img)
        if curr_mse < min_mse:
            min_mse = curr_mse
            trans_matrix = cand_trans_mat
    return trans_matrix


def mse(move_img, img2):
    return ((move_img - img2) ** 2).mean()


"""in each level in pyramid, find the translation matrix using LK
then find the rotation matrix using LK"""


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    # finding translation matrix                                      
    translation_mat = findTranslationLK(im1, im2)
    # finding rotation angle using harris corners detector
    angle_of_rotation = find_angle(im1, im2)
    """([[cos t, sin t, u],
        [-sin t, cos t, v],
        [0, 0, 1]])"""
    # making the rigid matrix
    translation_mat[0, 0] = math.cos(angle_of_rotation)
    translation_mat[0, 1] = math.sin(angle_of_rotation)
    translation_mat[1, 0] = -(math.sin(angle_of_rotation))
    translation_mat[1, 1] = math.cos(angle_of_rotation)

    return translation_mat


def find_angle(im1: ndarray, im2: ndarray) -> float:
    angle_of_rotation = 0
    min_mse = np.inf
    for angle__ in range(360):
        cand_rotation_mat = np.array([[math.cos(angle__), -math.sin(angle__), 0],
                                      [math.sin(angle__), math.cos(angle__), 0],
                                      [0, 0, 1]]).astype(float)
        warp_img = cv2.warpPerspective(im1, cand_rotation_mat, (im1.shape[1], im1.shape[0]))
        curr_mse = mse(im2, warp_img)
        if curr_mse < min_mse:
            min_mse = curr_mse
            angle_of_rotation = angle__
    return angle_of_rotation


def harris_corners_detector(img, k=0.05, win_size=5) -> list:
    """
    finding corners using harris corners detector algorithm
    :param img: image to find its corners
    :param k: sensitivity factor to separate corners from edges
    :param win_size: window which is shifted on the image
    :return: list with all corners
    """
    corners = []
    # threshold = 1000
    offset = win_size // 2
    height, width = img.shape[0], img.shape[1]
    # finding gradients
    # print(len(np.gradient(img)))
    dy, dx, z = np.gradient(img)
    xx = dx ** 2
    xy = dx * dy
    yy = dy ** 2
    for x in range(offset, height - offset):
        for y in range(offset, width - offset):
            sum_xx = np.sum(xx[x - offset:x + 1 + offset, y - offset:y + 1 + offset])
            sum_yy = np.sum(yy[x - offset:x + 1 + offset, y - offset:y + 1 + offset])
            sum_xy = np.sum(xy[x - offset:x + 1 + offset, y - offset:y + 1 + offset])
            # finding Harris response- determinant, trace and R
            determinant = (sum_xx * sum_yy) - (sum_xy ** 2)
            trace = sum_xx + sum_yy
            R = determinant - k * (trace ** 2)
            # finding if it is a corner
            if R > 0:
                corners.append([x, y])
    return corners


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    # finding corners
    # corners_im1 = harris_corners_detector(im1)
    # corners_im2 = harris_corners_detector(im2)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    """ no need to convert because images already in gray scale"""
    dest_img1 = cv2.cornerHarris(src=im1, blockSize=2, ksize=3, k=0.04)
    dest_img2 = cv2.cornerHarris(src=im2, blockSize=2, ksize=3, k=0.04)

    corners_im1 = im1[dest_img1 > 0.01 * dest_img1.max()]
    corners_im2 = im2[dest_img2 > 0.01 * dest_img2.max()]

    # finding deltas
    im1_x, im1_y = corners_im1[0], corners_im1[1]
    im2_x, im2_y = corners_im2[0], corners_im2[1]
    dx = im2_x / im1_x
    dy = im2_y / im1_y
    return np.asarray([[1, 0, dx],
                       [0, 1, dy],
                       [0, 0, 1]])


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    # finding translation matrix and angle of rotation
    translation_mat = findTranslationCorr(im1, im2)
    angle_of_rotation = find_angle(im1, im2)
    # assign angle in translation matrix making it rigid
    translation_mat[0][0] = math.cos(angle_of_rotation)
    translation_mat[0][1] = math.sin(angle_of_rotation)
    translation_mat[1][0] = -(math.sin(angle_of_rotation))
    translation_mat[1][1] = math.cos(angle_of_rotation)
    return translation_mat


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    # no need to check if images are in grayscale according to instructions
    warped_img = np.zeros(im2.shape)
    height, width = im2.shape[0], im2.shape[1]
    t_inverse = np.linalg.inv(T)
    # moving across all pixels on img2 and multiply them by the inverse of T to get the warped pixel
    for i in range(height):
        for j in range(width):
            curr_index = np.array([i, j, 1])
            new_index = np.dot(t_inverse, curr_index).astype(int)
            new_index_x, new_index_y = new_index[0], new_index[1]
            # making sure we don't exceed from the shape of img1
            if 0 <= new_index_x < im1.shape[0] and 0 <= new_index_y < im1.shape[1]:
                warped_img[i, j] = im1[new_index_x, new_index_y]

    plt.imshow(warped_img)
    plt.title("warped image")
    plt.show()
    return warped_img


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    # Each level in the pyramids, the image shape is cut in half, so for x levels, crop the initial image to 2x ·
    # ⌊imgsize/2x⌋
    height = int(img.shape[0] / 2 ** levels) * (2 ** levels)
    width = int(img.shape[1] / (2 ** levels)) * (2 ** levels)
    img = cv2.resize(img, (width, height)).astype(float)  # resizing img to format that can be divided by 2 x times
    pyramid = [img]  # adding original image to the pyramid
    img_copy = img.copy()
    for i in range(1, levels):  # adding each level to the list from biggest to smallest
        gauss_kernel = make_gauss_kernel()
        img_blur = cv2.filter2D(img_copy, -1, gauss_kernel)
        img_reduced = img_blur[::2, ::2]  # 2 times smaller
        img_copy = img_reduced
        pyramid.append(img_reduced)
    return pyramid


def make_gauss_kernel():
    # making gauss kernel with window size of 5
    sigma = int(round(0.3 * ((4 - 1) * 0.5 - 1) + 0.8))
    gauss = cv2.getGaussianKernel(5, sigma=sigma)
    gauss_kernel = gauss.dot(gauss.T)
    return gauss_kernel


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaussian_pyr = gaussianPyr(img, levels)
    gauss_kernel = make_gauss_kernel() * 4
    is_rgb = True if len(img.shape) == 3 else False
    for i in range(levels - 1):
        next_level_img = gaussian_pyr[i + 1]
        if is_rgb:
            expand = np.zeros(((next_level_img.shape[0] * 2), (next_level_img.shape[1] * 2), 3))
        else:  # grayscale
            expand = np.zeros((next_level_img.shape[0] * 2, next_level_img.shape[1] * 2))
        expand[::2, ::2] = next_level_img
        expand_img = cv2.filter2D(expand, -1, gauss_kernel)
        gaussian_pyr[i] = gaussian_pyr[i] - expand_img
    return gaussian_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    orig_img = lap_pyr[-1]  # the smallest image (from the gaussPyramid)
    gauss_kernel = make_gauss_kernel() * 4
    is_rgb = True if len(orig_img.shape) == 3 else False
    for i in range(levels - 1, 0, -1):
        if is_rgb:
            expand = np.zeros(((orig_img.shape[0] * 2), (orig_img.shape[1] * 2), 3))
        else:
            expand = np.zeros((orig_img.shape[0] * 2, orig_img.shape[1] * 2))
        expand[::2, ::2] = orig_img
        expand_img = cv2.filter2D(expand, -1, gauss_kernel)
        orig_img = expand_img + lap_pyr[i - 1]
    return orig_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    # resizing to mask shape in to be able to make operations
    img_1 = cv2.resize(img_1, (mask.shape[1], mask.shape[0]))
    img_2 = cv2.resize(img_2, (mask.shape[1], mask.shape[0]))
    # making pyramids
    lap_pyr1 = laplaceianReduce(img_1, levels)
    lap_yr2 = laplaceianReduce(img_2, levels)
    gauss_pyr = gaussianPyr(mask, levels)
    blend_img = (lap_pyr1[levels - 1] * gauss_pyr[levels - 1] + (1 - gauss_pyr[levels - 1]) * lap_yr2[levels - 1])
    gauss_kernel = make_gauss_kernel() * 4
    # expanding and blending
    for i in range(levels - 2, -1, -1):
        if len(img_1.shape) == 3:  # RGB img
            expand = np.zeros(((blend_img.shape[0] * 2), (blend_img.shape[1] * 2), 3))
        else:
            expand = np.zeros((blend_img.shape[0] * 2, blend_img.shape[1] * 2))
        expand[::2, ::2] = blend_img
        expand_img = cv2.filter2D(expand, -1, gauss_kernel)
        blend_img = expand_img + lap_pyr1[i] * gauss_pyr[i] + (1 - gauss_pyr[i]) * lap_yr2[i]
    naive_blend = img_1 * mask + img_2 * (1 - mask)
    naive_blend = cv2.resize(naive_blend, (blend_img.shape[1], blend_img.shape[0]))
    return naive_blend, blend_img
