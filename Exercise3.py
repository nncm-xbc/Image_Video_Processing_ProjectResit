import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import skimage.morphology

# import both images in grayscale
bubbles = cv2.imread("images/granulometry1-min.jpg", 0)
balls = cv2.imread("images/granulometry2-min.jpg", 0)


def granolumetry(img):

    # Successive opening operations
    granule = 0

    # Compute surface area
    return granule


granule1 = granolumetry(bubbles)
granule2 = granolumetry(balls)

plt.subplot(121), plt.imshow(granule1, cmap='gray')
plt.title('granulometry of bubbles image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(granule2, cmap='gray')
plt.title('granulometry of balls image'), plt.xticks([]), plt.yticks([])
plt.show()
