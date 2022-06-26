import numpy as np
import matplotlib.pyplot as plt


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    disparity_map = np.zeros(img_l.shape)
    hight, width = img_l.shape[0], img_l.shape[1]
    for i in range(k_size, hight - k_size):
        for j in range(k_size, width - k_size):
            left = img_l[i - k_size:i + k_size + 1, j - k_size:j + k_size + 1]
            min_ssd = np.inf
            for offset in range(disp_range[0], disp_range[1]):
                right_r = img_r[i - k_size:i + k_size + 1, j - k_size - offset:j + k_size + 1 + offset]
                right_l = img_r[i - k_size:i + k_size + 1, j - k_size - offset:j + k_size + 1 - offset]
                if right_r.shape == left.shape:
                    curr_ssd = ((left - right_r) ** 2).sum()
                    if curr_ssd < min_ssd:
                        disparity_map[i, j] = offset
                        min_ssd = curr_ssd
                if right_l.shape == left.shape:
                    curr_ssd = ((left - right_l) ** 2).sum()
                    if curr_ssd < min_ssd:
                        disparity_map[i, j] = offset
                        min_ssd = curr_ssd
    return disparity_map


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    disparity_map = np.zeros(img_l.shape)
    hight, width = img_l.shape[0], img_l.shape[1]
    for i in range(k_size, hight - k_size):
        for j in range(k_size, width - k_size):
            left = img_l[i - k_size:i + k_size + 1, j - k_size:j + k_size + 1]
            max_ncc = -np.inf
            for offset in range(disp_range[0], disp_range[1]):
                right_r = img_r[i - k_size:i + k_size + 1, j - k_size - offset:j + k_size + 1 + offset]
                right_l = img_r[i - k_size:i + k_size + 1, j - k_size - offset:j + k_size + 1 - offset]
                if right_r.shape == left.shape:
                    curr_ncc = normalised_cross_correlation(left, right_r)
                    if curr_ncc > max_ncc:
                        disparity_map[i, j] = offset
                        max_ncc = curr_ncc
                if right_l.shape == left.shape:
                    curr_ncc = normalised_cross_correlation(left, right_l)
                    if curr_ncc > max_ncc:
                        disparity_map[i, j] = offset
                        max_ncc = curr_ncc
    return disparity_map


def normalised_cross_correlation(roi, target):
    # Normalised Cross Correlation Equation
    cor = np.sum(roi * target)
    nor = np.sqrt((np.sum(roi ** 2))) * np.sqrt(np.sum(target ** 2))

    return cor / nor


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    A = []
    for i in range(len(src_pnt)):
        A.append([src_pnt[i][0], src_pnt[i][1], 1, 0, 0, 0, -(dst_pnt[i][0]) * src_pnt[i][0],
                  -(dst_pnt[i][0]) * src_pnt[i][1], -(dst_pnt[i][0])])
        A.append([0, 0, 0, src_pnt[i][0], src_pnt[i][1], 1, -(dst_pnt[i][1]) * src_pnt[i][0],
                  -(dst_pnt[i][1]) * src_pnt[i][1], -(dst_pnt[i][1])])
    u, d, v = np.linalg.svd(np.array(A))  # need only vh
    matrix = (v[-1] / v[-1, -1]).reshape(3, 3)
    src_pnt = np.hstack((src_pnt, np.ones((src_pnt.shape[0], 1)))).T
    h_src = matrix.dot(src_pnt)
    h_src /= h_src[2, :]
    return matrix, np.sqrt(np.sum(h_src[0:2, :] - dst_pnt.T) ** 2)


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """
    dst_p = []
    src_p = []

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    fig1 = plt.figure()
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######
    # display image 2
    fig2 = plt.figure()
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    matrix, error = computeHomography(src_pnt=src_p, dst_pnt=dst_p)

    highet, width = src_img.shape[0], src_img.shape[1]
    blend_image = np.zeros(dst_img.shape)
    for i in range(highet):
        for j in range(width):
            new_p = np.dot(matrix, np.array([j, i, 1]).T)
            new_p /= new_p[2]
            # if the values are out of bounds, do nothing
            if 0 <= int(new_p[1]) < highet and 0 <= int(new_p[0]) < width:
                blend_image[int(new_p[1]), int(new_p[0])] = src_img[i, j]

    error = blend_image == 0
    canvas = dst_img * error + (1 - error) * blend_image
    plt.imshow(canvas)

    plt.show()
