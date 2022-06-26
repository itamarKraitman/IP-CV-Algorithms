import cv2
import matplotlib.pyplot as plt
import numpy as np

from ex3_utils import *
import time
import warnings

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """

    """plotting inside function!!!"""

    print("Hierarchical LK Demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    uv = opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), k=6, stepSize=20, winSize=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    # median_hirr = np.median(uv, 0)
    # mean_hirr = np.mean(uv, 0)

    # displayOpticalFlow(img_2, pts, uv)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    pts_lk, uv_lk = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    uv_hirr = opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), k=6, stepSize=20, winSize=5)

    # showing mean and median differences
    median_lk = np.median(uv_lk, 0)
    mean_lk = np.mean(uv_lk, 0)
    median_hirr = np.median(uv_hirr.reshape(uv_hirr.shape[1], uv_hirr.shape[0]), 0)
    mean_hirr = np.mean(uv_hirr.reshape(uv_hirr.shape[1], uv_hirr.shape[0]), 0)

    print("mean difference: ", abs(mean_hirr - mean_lk))
    print("median difference: ", abs(median_hirr - median_lk))

    # I did not fina a way to save the plot as image (imwrite and figsave don't help- looking bad),
    # so I saved a screen shoot of the plot
    lk_pyr_img = cv2.cvtColor(cv2.imread("images/boxMan_compare.jpg"), cv2.COLOR_BGR2RGB)

    # plotting results side by side
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(img_1, cmap='gray')
    axis[0].quiver(pts_lk[:, 0], pts_lk[:, 1], uv_lk[:, 0], uv_lk[:, 1], color='r')
    axis[0].set_title("LK Iterative")
    axis[1].imshow(lk_pyr_img)
    axis[1].set_title("hierarchical LK")
    plt.suptitle("LK compare")

    plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")
    img1 = cv2.cvtColor(cv2.imread("images/test_img.jpg"), cv2.COLOR_BGR2GRAY)

    t_trans = np.array([[1, 0, -2],
                        [0, 1, -4],
                        [0, 0, 1]]).astype(float)
    #   angle = 50
    t_rigid = np.array([[0.6427876097, -0.7660444431, -2],
                        [0.7660444431, 0.6427876097, -4],
                        [0, 0, 1]]).astype(float)
    cv2_img_trans = cv2.warpPerspective(img1, t_trans, (img1.shape[1], img1.shape[0]))
    cv2_img_rigid = cv2.warpPerspective(img1, t_rigid, (img1.shape[1], img1.shape[0]))
    translation_lk_show(img1, cv2_img_trans)
    rigid_lk_show(img1, cv2_img_rigid)
    translation_corr_show(img1, cv2_img_trans)
    rigid_corr_show(img1, cv2_img_rigid)
    warp_test(img1, t_trans)


def translation_lk_show(img1, cv2_img):
    print("translation Luckas Kanade")
    st = time.time()
    trans_lk = findTranslationLK(img1, cv2_img)
    et = time.time()
    img_lk = cv2.warpPerspective(img1, trans_lk, (img1.shape[1], img1.shape[0]))
    fig, axis = plt.subplots(1, 2)
    print("Time: {:.4f}".format(et - st))

    axis[0].imshow(cv2_img)
    axis[0].set_title("cv2")
    axis[1].imshow(img_lk)
    axis[1].set_title("my implementation")
    plt.suptitle("translation LK")

    cv2.imwrite(filename="images/imTransA1.jpg", img=img_lk)

    plt.show()


def rigid_lk_show(img1, cv2_img):
    print("rigid Luckas Kanade")
    st = time.time()
    trans_lk = findRigidLK(cv2_img, img1)
    et = time.time()
    img_lk = cv2.warpPerspective(img1, trans_lk, (img1.shape[1], img1.shape[0]))
    fig, axis = plt.subplots(1, 2)
    print("Time: {:.4f}".format(et - st))

    axis[0].imshow(cv2_img)
    axis[0].set_title("cv2")
    axis[1].imshow(img_lk)
    axis[1].set_title("my implementation")
    plt.suptitle("rigid LK")

    cv2.imwrite(filename="images/imRigidA1.jpg", img=img_lk)

    plt.show()


def translation_corr_show(img1, cv2_img):
    print("translation correlation")
    st = time.time()
    trans_corr = findTranslationCorr(img1, cv2_img)
    et = time.time()
    img_corr = cv2.warpPerspective(img1, trans_corr, (img1.shape[1], img1.shape[0]))
    fig, axis = plt.subplots(1, 2)
    print("Time: {:.4f}".format(et - st))

    axis[0].imshow(cv2_img)
    axis[0].set_title("cv2")
    axis[1].imshow(img_corr)
    axis[1].set_title("my implementation")
    plt.suptitle("translation correlation")

    cv2.imwrite(filename="images/imTransA2.jpg", img=img_corr)

    plt.show()


def rigid_corr_show(img1, cv2_img):
    print("rigid correlation")
    st = time.time()
    rigid_img = findRigidCorr(cv2_img, img1)
    et = time.time()
    img_rigid = cv2.warpPerspective(img1, rigid_img, (img1.shape[1], img1.shape[0]))
    fig, axis = plt.subplots(1, 2)
    print("Time: {:.4f}".format(et - st))

    axis[0].imshow(cv2_img)
    axis[0].set_title("cv2")
    axis[1].imshow(img_rigid)
    axis[1].set_title("my implementation")
    plt.suptitle("rigid correlation")

    cv2.imwrite(filename="images/imRigidA2.jpg", img=img_rigid)

    plt.show()


def warp_test(img: ndarray, t: ndarray):
    print("warping test")
    img_warp_cv2 = cv2.warpPerspective(img, t, (img.shape[1], img.shape[0]))
    st = time.time()
    warp_img = warpImages(im1=img, im2=img_warp_cv2, T=t)
    et = time.time()
    fig, axis = plt.subplots(1, 2)
    print("Time: {:.4f}".format(et - st))

    axis[0].imshow(img)
    axis[0].set_title("original image")
    axis[1].imshow(warp_img)
    axis[1].set_title("warped image")
    plt.suptitle("warp test")

    plt.show()


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        # print(lv_idx)
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('images/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('images/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('images/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'images/boxMan.jpg'
    # lkDemo(img_path)
    # hierarchicalkDemo(img_path)
    compareLK(img_path)

    # imageWarpingDemo(img_path)

    # pyrGaussianDemo('images/pyr_bit.jpg')
    # pyrLaplacianDemo('images/pyr_bit.jpg')
    # blendDemo()


if __name__ == '__main__':
    main()
