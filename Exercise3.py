import cv2
from matplotlib import pyplot as plt
import skimage.morphology


def granolumetry(img):

    e_values = [10, 20, 25, 30]
    # Successive opening operations
    for e in e_values:
        print("Disk radii = ", e)
        # perform morphological opening with increasing structuring element,
        # here a disk.
        img = skimage.morphology.dilation(img, skimage.morphology.disk(e))

        # Compute surface area
        # compute the sum of all the pixels of the image after each opening or only the pixels affected by the operation ?
        surface_area = sum(sum(img))
        print("Surface area: ", surface_area)

        plt.subplot(111), plt.imshow(img, cmap='gray')
        plt.title('image smoothed'), plt.xticks([]), plt.yticks([])
        plt.show()

    return img


# import both images in grayscale
bubbles = cv2.imread("images/granulometry1-min.jpg", 0)
balls = cv2.imread("images/granulometry2-min.jpg", 0)

print("-----Granulometry of the bubbles picture-----")
granule1 = granolumetry(bubbles)
print("-----Granulometry of the balls picture-----")
granule2 = granolumetry(balls)
