import cv2
import numpy as np
from scipy.signal.signaltools import wiener
from matplotlib import pyplot as plt


""""
-----------Question 2-----------
"""


# function that computes a motion blur mask given an image
def motion_blur(img, kernel):
    """
    This funtion first creates a kernel that will be applied to the image for the blur.
    The size of the kernel decides how much motion blur is applied to the image.
    The motion blur applied in this funtion is a  horizontal motion blur.
    This is done by filling the middle row of the kernel with ones.
    The kernel is then normalized to allow the sum of the orignal image and the noise filter,
    which is then computed and outputed.
    :param img: Input image on which we want to apply the box filter
    :param kernel: kernel that will be applied to the image
    :return: blurred version of the original image
    """

    # size of the kernel
    kernel_size = kernel.shape[:1][0]

    # fill in the horizontal middle row of the kernel for a horizontal motion blur
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # normalize the mask\kernel by its size
    kernel /= kernel_size

    # apply the kernel or mask to the given image using filter2D
    blurred_img = cv2.filter2D(img, -1, kernel)

    return blurred_img


# import image in grayscale
image = cv2.imread("images/flowers-min.jpg", 0)

# Create the kernel\mask that will be applied to the given image.
# the larger the kernel, the more motion is applied to the image
kernel = np.zeros((100, 100))

# apply the blur to the imported image
blurred_image = motion_blur(image, kernel)

# apply a gaussian noise to the blurred image
# since the gaussian noise follows the standard deviation we use np.random.normal()
noisy_img = blurred_image + np.random.normal(0, 10, blurred_image.shape)

# plot the blurred image and the noisy image
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Initial image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(blurred_image, cmap='gray')
plt.title('Motion blur image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy image'), plt.xticks([]), plt.yticks([])
plt.show()

# compute the FFT of the final image
fft_original = np.fft.fft2(image)
fft_noisy = np.fft.fft2(noisy_img)

# shift both fft for better understanding when plotting
shiftfft_original = np.fft.fftshift(fft_original)
shiftfft2_noisy = np.fft.fftshift(fft_noisy)

# plot the fft of the original image and the noisy one
plt.subplot(121), plt.imshow(np.log(1+np.abs(shiftfft_original)), cmap='gray')
plt.title('FFT of original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.log(1+np.abs(shiftfft2_noisy)), cmap='gray')
plt.title('FFT of noisy image'), plt.xticks([]), plt.yticks([])
plt.show()


""""
-----------Question 3-----------
"""


def inverse_filter(img):
    """
    This function implements inverse filtering. The first step is to translate the image into frequency domain.
    Then the given image is normalized to [0, 1].
    The damage of the original image is then computed by multiplying the normalized image and its FFT
    :param img: noisy or damages image
    :return: reversed noisy image
    """

    # FFT of image to able to process the image in the frequency domain
    fft_img = np.fft.fft2(img)
    shift_fft = np.fft.fftshift(fft_img)

    # normalize to [0,1]
    normalized_img = img / 255.

    # compute the "damage" of the image
    damaged_img = normalized_img * shift_fft

    # Apply inverse Filter
    inverse_img = damaged_img / normalized_img

    # Inverse FFT to return an image in the spatial domain
    inverse = np.fft.ifft2(np.fft.ifftshift(inverse_img))

    return inverse


def wiener_filter(img, kernel):
    kernel /= np.sum(kernel)

    # create an oputput copy of the image and compute its FFT
    output_copy = np.copy(img)
    output_fft = np.fft.fft2(output_copy)

    # compute complex conjugate
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel_inverse = 1/kernel * (np.abs(np.conj(kernel) ** 2) / (np.abs(kernel) ** 2 + 25))

    # apply the inverse kernel and compute inverse FFT for spatial domain image output
    output_fft = output_fft * kernel_inverse
    output = np.abs(np.fft.ifft2(output_fft))

    return output


# apply the wiener filtering or MMSE
MMSE_img = wiener_filter(noisy_img, kernel)

# plot the MMSE image
plt.subplot(111), plt.imshow(np.log(1+np.abs(MMSE_img)), cmap='gray')
plt.title('MMSE_img of noisy image'), plt.xticks([]), plt.yticks([])
plt.show()

# apply the inverse filter on the previous noisy image
reversed_img = inverse_filter(noisy_img)

# plot the reversed image
plt.subplot(111), plt.imshow(np.log(1+np.abs(reversed_img)), cmap='gray')
plt.title('Reverse of noisy image'), plt.xticks([]), plt.yticks([])
plt.show()

