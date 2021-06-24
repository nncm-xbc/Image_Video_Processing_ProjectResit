import cv2
import numpy as np
from matplotlib import pyplot as plt

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


# compute FFT of modified images
fft_bricks2 = np.fft.fft2(bricks)
fft_flowers2 = np.fft.fft2(flowers)
fft_grass2 = np.fft.fft2(grass)
fft_lamps2 = np.fft.fft2(lamps)

# shift them for more readability
shiftfft_bricks = np.fft.fftshift(fft_bricks)
shiftfft_flowers = np.fft.fftshift(fft_flowers)
shiftfft_grass = np.fft.fftshift(fft_grass)
shiftfft_lamps = np.fft.fftshift(fft_lamps)

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
ax1.imshow(np.log(1+np.abs(shiftfft_bricks)), cmap='gray')
ax2 = fig2.add_subplot(2, 2, 2)
ax2.imshow(np.log(1+np.abs(shiftfft_flowers)), cmap='gray')
ax3 = fig2.add_subplot(2, 2, 3)
ax3.imshow(np.log(1+np.abs(shiftfft_grass)), cmap='gray')
ax4 = fig2.add_subplot(2, 2, 4)
ax4.imshow(np.log(1+np.abs(shiftfft_lamps)), cmap='gray')
fig2.show()

"""
---------Question 3---------
"""

def power_spectrum(fft_img):

    # get its real and imaginary part
    img_real = np.real(fft_img)
    img_imag = np.imag(fft_img)

    # compute power spectrum
    power = img_real**2 + img_imag**2
    power_check = np.abs(fft_img)**2

    print("Power spectrum 1:\n", power)
    print("Power spectrum 2:\n", power_check)
    return power_check


# compute their power spectrum
# get their real and imaginary parts:
power_bricks = power_spectrum(fft_bricks)
power_flowers = np.fft.fftshift(power_spectrum(fft_flowers))
power_grass = np.fft.fftshift(power_spectrum(fft_grass))
power_lamps = np.fft.fftshift(power_spectrum(fft_lamps))

fig3 = plt.figure()
ax1 = fig3.add_subplot(2, 2, 1)
ax1.imshow(np.log(1+np.abs(power_bricks)), cmap='gray')
ax2 = fig3.add_subplot(2, 2, 2)
ax2.imshow(np.log(1+np.abs(power_flowers)), cmap='gray')
ax3 = fig3.add_subplot(2, 2, 3)
ax3.imshow(np.log(1+np.abs(power_grass)), cmap='gray')
ax4 = fig3.add_subplot(2, 2, 4)
ax4.imshow(np.log(1+np.abs(power_lamps)), cmap='gray')
fig3.show()
