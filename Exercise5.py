import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

bricks = cv2.imread("images/bricks-min.jpg", 0)
flowers = cv2.imread("images/flowers-min.jpg", 0)
grass = cv2.imread("images/grass-min.jpg", 0)
lamps = cv2.imread("images/lamps-min.jpg", 0)

"""
---------Question 1---------
"""
# compute FFT of all 4 images
fft_bricks = np.fft.fft2(bricks)
fft_flowers = np.fft.fft2(flowers)
fft_grass = np.fft.fft2(grass)
fft_lamps = np.fft.fft2(lamps)

# change periodicity of the 4 original images
bricks2 = cv2.imread("images/bricks-min2.jpg", 0)
flowers2 = cv2.imread("images/flowers-min2.jpg", 0)
grass2 = cv2.imread("images/grass-min2.jpg", 0)
lamps2 = cv2.imread("images/lamps-min2.jpg", 0)

# compute FFT of modified images
fft_bricks2 = np.fft.fft2(bricks2)
fft_flowers2 = np.fft.fft2(flowers2)
fft_grass2 = np.fft.fft2(grass2)
fft_lamps2 = np.fft.fft2(lamps2)

# shift them for more readability
shiftfft_bricks = np.fft.fftshift(fft_bricks)
shiftfft_flowers = np.fft.fftshift(fft_flowers)
shiftfft_grass = np.fft.fftshift(fft_grass)
shiftfft_lamps = np.fft.fftshift(fft_lamps)

# shift them for more readability
shiftfft_bricks2 = np.fft.fftshift(fft_bricks2)
shiftfft_flowers2 = np.fft.fftshift(fft_flowers2)
shiftfft_grass2 = np.fft.fftshift(fft_grass2)
shiftfft_lamps2 = np.fft.fftshift(fft_lamps2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 2, 1)
ax1.imshow(bricks, cmap='gray')
ax2 = fig1.add_subplot(2, 2, 2)
ax2.imshow(flowers, cmap='gray')
ax3 = fig1.add_subplot(2, 2, 3)
ax3.imshow(grass, cmap='gray')
ax4 = fig1.add_subplot(2, 2, 4)
ax4.imshow(lamps, cmap='gray')
fig1.show()

fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 2, 1)
ax1.imshow(bricks2, cmap='gray')
ax2 = fig2.add_subplot(2, 2, 2)
ax2.imshow(flowers2, cmap='gray')
ax3 = fig2.add_subplot(2, 2, 3)
ax3.imshow(grass2, cmap='gray')
ax4 = fig2.add_subplot(2, 2, 4)
ax4.imshow(lamps2, cmap='gray')
fig2.show()

fig3 = plt.figure()
ax1 = fig3.add_subplot(2, 2, 1)
ax1.imshow(np.log(1+np.abs(shiftfft_bricks)), cmap='gray')
ax2 = fig3.add_subplot(2, 2, 2)
ax2.imshow(np.log(1+np.abs(shiftfft_flowers)), cmap='gray')
ax3 = fig3.add_subplot(2, 2, 3)
ax3.imshow(np.log(1+np.abs(shiftfft_grass)), cmap='gray')
ax4 = fig3.add_subplot(2, 2, 4)
ax4.imshow(np.log(1+np.abs(shiftfft_lamps)), cmap='gray')
fig3.show()

fig4 = plt.figure()
ax1 = fig4.add_subplot(2, 2, 1)
ax1.imshow(np.log(1+np.abs(shiftfft_bricks2)), cmap='gray')
ax2 = fig4.add_subplot(2, 2, 2)
ax2.imshow(np.log(1+np.abs(shiftfft_flowers2)), cmap='gray')
ax3 = fig4.add_subplot(2, 2, 3)
ax3.imshow(np.log(1+np.abs(shiftfft_grass2)), cmap='gray')
ax4 = fig4.add_subplot(2, 2, 4)
ax4.imshow(np.log(1+np.abs(shiftfft_lamps2)), cmap='gray')
fig4.show()

"""
---------Question 3---------
"""


def power_spectrum(fft_img):

    # get the image's real and imaginary part
    img_real = np.real(fft_img)
    img_imag = np.imag(fft_img)

    # compute power spectrum
    power = img_real**2 + img_imag**2
    power_check = np.abs(fft_img)**2

    # ("Power spectrum 1:\n", power)
    # print("Power spectrum 2:\n", power_check)
    return power


def entropy(img, vectorized_img):
    # normalize power spectrum values
    normalized_power = normalize(vectorized_img[:,np.newaxis], axis=0).ravel()

    # compute entropy
    (m, n) = img.shape
    h = 0
    for j in range(m * n):
        h += normalized_power[j]*np.log(normalized_power[j])

    return h


# compute their power spectrum
power_bricks = power_spectrum(fft_bricks)
power_flowers = np.fft.fftshift(power_spectrum(fft_flowers))
power_grass = np.fft.fftshift(power_spectrum(fft_grass))
power_lamps = np.fft.fftshift(power_spectrum(fft_lamps))

# vectorize the power spectrum
vectorized_bricks = power_bricks.flatten()
vectorized_flowers = power_flowers.flatten()
vectorized_grass = power_grass.flatten()
vectorized_lamps = power_lamps.flatten()

print(vectorized_bricks)
print(vectorized_flowers)
print(vectorized_grass)
print(vectorized_lamps)

print("Bricks entropy:\n", entropy(bricks, vectorized_bricks))
print("Flowers entropy:\n", entropy(flowers, vectorized_flowers))
print("Grass entropy:\n", entropy(grass, vectorized_grass))
print("Lamps entropy:\n", entropy(lamps, vectorized_lamps))

