import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

""""
-----------Question 1-----------
"""

def convolution(img, kernel):
    """
    Given an image and a filter this function performs a convolution of the two arguments.
    In details the function applies the given box filter to each pixel of the original image.
    Each pixel of the original image is considered the center of a matrix of the size of the filter.
    This allows to multiply the filter and the small matrix. Then the sum of all the values at the same position.
    This gives us a value for the processed pixel at the same position as the original pixel.
    Credits : https://github.com/ashushekar/image-convolution-from-scratch
    :param img: Input image on which we want to apply the box filter
    :param filter: box filter which we want to apply to the given image
    :return: modified version of the original image
    """

    # create output matrix of the size of the input image
    output_img = np.zeros_like(img)

    # Add zero padding to the input image
    # Adding zeros to the output image to allow the matrix multiplication
    # to be made on the edge pixels of the original image
    img_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    img_padded[1:-1, 1:-1] = img

    # Go over every pixel of the image
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            # multiplication of the kernel and the image on a specific pixel
            output_img[y, x] = (kernel * img_padded[y: y+3, x: x+3]).sum()
            # print(output_img[y, x])

    return output_img


# import image in greyscale
img = cv2.imread('images/Fallen-Angel.jpg', 0)

# box filter 3x3 (kernel)
filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# call the convolution function on our image and box filter.
processed_img = convolution(img, filter)


# plot both original and processed images with convolution function
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('initial image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(processed_img, cmap='gray')
plt.title('processed image'), plt.xticks([]), plt.yticks([])
plt.show()

""""
-----------Question 3-----------
"""

fft_original_img = np.fft.fft2(img)
fft_filter = np.fft.fft2(filter)
fft_processed_img = np.fft.fft2(processed_img)

shiftfft_original = np.fft.fftshift(fft_original_img)
shiftfft_filter = np.fft.fftshift(fft_filter)
shiftfft_processed = np.fft.fftshift(fft_processed_img)

plt.subplot(231), plt.imshow(np.log(1+np.abs(shiftfft_original)), cmap='gray')
plt.title('FFT of initial image'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(np.log(1+np.abs(shiftfft_filter)), cmap='gray')
plt.title('FFT of the filter'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(np.log(1+np.abs(shiftfft_processed)), cmap='gray')
plt.title('FFT of processed image'), plt.xticks([]), plt.yticks([])
plt.show()
