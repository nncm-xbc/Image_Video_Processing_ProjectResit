import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def convolution(img, filter):
    """
    Given an image and a filter this function performs a convolution of the two arguments.
    :param img: Input image on which we want to apply the box filter
    :param filter: box filter which we want to apply to the given image
    :return: modified version of the original image
    """

    x, y = img.shape
   # loop through the image and apply the filter.
    for i in range(x):
        for j in range(y):
            img[i][j] = img[i][j] * filter


# import image in greyscale
f = cv2.imread('images/Fallen-Angel.jpg', 0)

# box filter 3x3
h = np.ones((3, 3), np.float32)

# call the convolution function on our image and box filter.
convolution(f, h)

plt.subplot(111), plt.imshow(f, cmap='gray')
plt.title('initial image'), plt.xticks([]), plt.yticks([])
plt.show()
