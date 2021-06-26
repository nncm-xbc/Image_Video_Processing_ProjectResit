import cv2
from matplotlib import pyplot as plt
import skimage.morphology
import numpy as np


def granolumetry(img):

    area_values = []

    e_values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # Successive opening operations
    for e in e_values:
        print("Disk radii = ", e)
        # perform morphological opening with increasing structuring element,
        # here a disk.
        img = skimage.morphology.opening(img, skimage.morphology.disk(e))

        # Compute surface area
        # compute the sum of all the pixels of the image after each opening or only the pixels affected by the operation ?
        surface_area = sum(sum(img))

        area_values.append(surface_area)

        plt.subplot(111), plt.imshow(img, cmap='gray')
        plt.title('image smoothed'), plt.xticks([]), plt.yticks([])
        plt.show()

    # compute the difference between adjacent elements of the area values array
    differences = []
    previous = 0
    for area in area_values:
        if area_values[0] == area:
            difference_val = 0
            differences.append(difference_val)
            previous = area
        else:
            difference_val = np.abs(area - previous)
            differences.append(difference_val)
            previous = area
    print("differences :", differences)

    return img, e_values, differences


# import both images in grayscale
bubbles = cv2.imread("images/granulometry1-min.jpg", 0)
balls = cv2.imread("images/granulometry2-min.jpg", 0)

print("-----Granulometry of the bubbles picture-----")
granule1, e_val_bubble, area_diff_bubble = granolumetry(bubbles)

plt.plot(e_val_bubble, area_diff_bubble)
plt.xlabel("Disk radius r")
plt.ylabel("Differences")
plt.show()

print("-----Granulometry of the balls picture-----")
granule2,  e_val_ball, area_diff_ball = granolumetry(balls)

plt.plot(e_val_ball, area_diff_ball)
plt.xlabel("Disk radius r")
plt.ylabel("Differences")
plt.show()


