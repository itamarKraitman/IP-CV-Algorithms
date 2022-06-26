import math
import numpy as np
import cv2


def myID():
    return 208925578


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    flip_kernel = np.flip(k_size)
    # init the new vector with the proper size
    result = np.zeros(in_signal.size + k_size.size - 1, dtype=int)
    # zeros padding
    zeros_padding = np.zeros(k_size.size - 1)
    new_signal = np.append(zeros_padding, np.append(in_signal, zeros_padding))
    # computing the result vector
    for i in range(result.size):
        result[i] = np.dot(new_signal[i: i + k_size.size], flip_kernel)
    # converting result vector to int vector in order to match np.convolve()
    return result


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    if kernel.max() > 1:
        kernel = cv2.normalize(kernel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    flip_kernel = np.flip(kernel)
    high_img, wid_img = in_image.shape
    high_ker, wid_ker = kernel.shape
    # zeros padding
    pad_img = np.pad(in_image, (math.floor(high_ker // 2), math.floor(wid_ker // 2)), 'edge')
    result = np.zeros((high_img, wid_img))
    # computing the convolution
    for i in range(high_img):
        for j in range(wid_img):
            result[i, j] = (pad_img[i: i + high_ker, j: j + wid_ker] * flip_kernel).sum()
    return result


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    if in_image.max() > 1:
        in_image = cv2.normalize(in_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # vector to perform the convolutions with
    x_conv = np.asarray([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    y_conv = np.asarray(x_conv).transpose()
    # compute derivatives
    r_der = conv2D(in_image=in_image, kernel=x_conv)
    c_der = conv2D(in_image=in_image, kernel=y_conv)
    # magnitude
    magnitude = np.sqrt(np.square(r_der) + np.square(c_der))
    # directions
    directions = np.arctan2(c_der, r_der) * 180 / np.pi
    return directions, magnitude


# bounus
def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    if in_image.max() > 1:
        in_image = cv2.normalize(in_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if k_size % 2 == 1:  # k_size should always be odd
        # sigma formula-https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#Mat%20getGaussianKernel(int%20ksize,%20double%20sigma,%20int%20ktype)
        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
        kernel = np.ndarray((k_size, k_size)).astype(int)
        # filling the kernel with the gaussian distribution
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                kernel[i][j] = ((1 / 2 * np.pi * np.square(sigma)) * np.e) - ((i ** 2 + j ** 2) / 2 * np.square(sigma))
        # print(kernel.sum())
        kernel = kernel
        return conv2D(in_image, kernel)
    else:
        print("kernel size should be an odd number")


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    if in_image.max() > 1:
        in_image = cv2.normalize(in_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if k_size % 2 == 1:
        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
        kernel = cv2.getGaussianKernel(ksize=k_size, sigma=sigma)
        return cv2.filter2D(src=in_image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    else:
        print("kernel size should be odd")


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    pass


# choose one of two
def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    if img.max() > 1:
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    """opencv solution"""
    # # smooting with 2d gaussian filter
    # gaus_img_ocv = cv2.GaussianBlur(img, (5, 5), 1)
    # # apply laplacian filter
    # lap_img_ocv = cv2.Laplacian(gaus_img_ocv, cv2.CV_64F)
    # lap_img_ocv_final = lap_img_ocv / lap_img_ocv.max()
    """my solution"""
    blur_img = blurImage2(in_image=img, k_size=5)
    # blur_img = cv2.GaussianBlur(img, (5,5), 0)
    lap_img = conv2D(blur_img, laplacian)
    my_sol_img = np.zeros(img.shape)
    l_rows, l_cols = laplacian.shape
    img_rows, img_cols = lap_img.shape
    # zero crossing
    for i in range(img_rows - (l_rows - 1)):
        for j in range(img_cols - (l_cols - 1)):
            if lap_img[i][j] == 0:
                # checking all neighbours
                if (lap_img[i][j - 1] < 0 and lap_img[i][j + 1] > 0) or \
                        (lap_img[i][j - 1] < 0 and lap_img[i][j + 1] < 0) or \
                        (lap_img[i - 1][j] < 0 and lap_img[i + 1][j] > 0) or \
                        (lap_img[i - 1][j] > 0 and lap_img[i + 1][j] < 0):
                    my_sol_img[i][j] = 255
            if lap_img[i][j] < 0:
                if (lap_img[i][j - 1] > 0) or (lap_img[i][j + 1] > 0) or (lap_img[i - 1][j] > 0) or (
                        lap_img[i + 1][j] > 0):
                    my_sol_img[i][j] = 255
    return my_sol_img


# def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
#     if min_radius <= 0 or max_radius <= 0 or min_radius >= max_radius:
#         print("input error")
#         return []
#     if img.max() > 1:
#         img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     circles = []
#     radius = max_radius - min_radius
#     # Blurring the image and find its edges
#     img_blur = cv2.GaussianBlur(src=img, ksize=(5, 5), sigmaX=1)
#     img_blur = img_blur.astype(np.uint8)
#     img_canny = cv2.Canny((img_blur * 255).astype(np.uint8), img_blur.shape[0], img_blur.shape[1])
#     print(img_canny.max())
#     # voting process
#     rows, cols = img_canny.shape
#     theta = np.arange(0, 360) * np.pi / 180
#     for r in range(round(min_radius), round(max_radius) + 1):
#         acc_array = np.zeros(img_canny.shape)
#         for x in range(rows):
#             for y in range(cols):
#                 if img_canny[x, y] == 255:  # edge
#                     for t in theta:
#                         b = round(y - r * np.sin(t))
#                         a = round(x - r * np.cos(t))
#                         if 0 <= a < rows and 0 <= b < cols:
#                             # print(1)
#                             acc_array[a, b] += 1
#         # finding circles with current radius
#         acc_max = np.max(acc_array)
#         # threshold = acc_max / 2
#         threshold = 10
#         if acc_max > threshold:
#             acc_array[acc_array < 0] = 0
#             for i in range(1, rows - 1):
#                 for j in range(1, cols - 1):
#                     if acc_array[i, j] >= threshold:
#                         # avg = np.float((
#                         #                        acc_array[i][j] + acc_array[i - 1][j] + acc_array[i + 1][j] +
#                         #                        acc_array[i][j - 1] +
#                         #                        acc_array[i][j + 1] + acc_array[i - 1][j - 1] + acc_array[i - 1][j + 1] +
#                         #                        acc_array[i + 1][
#                         #                            j - 1] + acc_array[i + 1][j + 1]) / 9)
#                         avg = acc_array[i - 1: i + 2, j - 1: j + 2].sum() / 9
#                         if avg >= threshold / 9:
#                             circles.append((i, j, r))
#                             acc_array[acc_array == avg] = 0
#     print(circles)
#     return circles

def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    # normalizing
    if img.max() <= 1:
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # computing derivatives
    x = cv2.Sobel(img, cv2.CV_64F, 0, 1, 0.7)
    y = cv2.Sobel(img, cv2.CV_64F, 1, 0, 0.7)
    direction = np.radians(np.arctan2(x, y) * 180 / np.pi)
    img_edges = cv2.Canny(img.astype(np.uint8), 50, 100) / 255
    acc_arr = np.zeros((img.shape[0], img.shape[1], max_radius + 1))
    rows_edges, cols_edges = img_edges.shape
    acc_rows, acc_cols = acc_arr.shape[0], acc_arr.shape[1]
    for x in range(0, rows_edges):
        for y in range(0, cols_edges):
            if img_edges[x, y] > 0:
                # for each r, computing the points of the circle, and vote in acc if the center is in its range
                for r in range(min_radius, max_radius + 1):
                    angle = direction[x, y] - np.pi / 2
                    x1 = int(x - r * math.cos(angle))
                    y1 = int(y + r * math.sin(angle))
                    if 0 < x1 < acc_rows and 0 < y1 < acc_cols:
                        acc_arr[x1, y1, r] += 1
                    x2 = int(x + r * math.cos(angle))
                    y2 = int(y - r * math.sin(angle))
                    if 0 < x2 < acc_rows and 0 < y2 < acc_cols:
                        acc_arr[x2, y2, r] += 1
    # picking threshold (I tried some, this is the best!)
    threshold = acc_arr.max() / 2
    b, a, radius = np.where(acc_arr >= threshold)
    # making the circles list
    circles = list()
    for i in range(0, len(a)):
        if a[i] == 0 and b[i] == 0 and radius[i] == 0:
            continue
        circles.append((a[i], b[i], radius[i]))
    print("best threshold is max value of accumulator / 2")
    if len(circles) == 0:
        print("no circles in image")
    return circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    rows, cols = in_image.shape
    new_image = np.zeros((rows, cols))
    half_kernel = int(np.ceil(k_size / 2))
    for i in range(rows):
        for j in range(cols):
            weights = 0
            result = 0
            # for each pixel, computing the new intensity according the formula
            for x in range(i - half_kernel, i + half_kernel):
                for y in range(j - half_kernel, j + half_kernel):
                    if 0 <= x < rows and 0 <= y < cols:
                        # gaussian smoothing
                        distance = math.sqrt(((i - x) ** 2) + ((j - y) ** 2))
                        gauss = math.exp(-distance / (2 * (sigma_color ** 2)))
                        # color similarity factor
                        color_similarity = (abs(int(in_image[i][j]) - int(in_image[x][y]))) ** 2
                        similarity_factor = math.exp(-color_similarity / (2 * (sigma_space ** 2)))
                        # computing new intensity
                        norm = gauss * similarity_factor
                        neighbor = in_image[x, y]
                        result += neighbor * norm
                        weights += norm
            # setting the new intensity into the filtered image
            result = result / weights
            new_image[i][j] = int(round(result))
    cv2_result = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    return cv2_result, new_image
