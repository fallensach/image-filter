import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal, datasets
from matplotlib.colors import LogNorm
import cv2

hp_filter = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]
                      ]) / 8

lp_filter = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]
                      ]) / 16

sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]
                    ]) / 8

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
                    ) / 8


def plot_spectrum(im_fft):
    plt.imshow(np.abs(im_fft), plt.cm.gray, norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.show()


def plot_outline(img):
    d_x = signal.convolve2d(img, sobel_x)
    d_xy = signal.convolve2d(d_x, sobel_y)
    # outlined_img = fftpack.ifft2(d_xy)
    plt.imshow(np.abs(d_xy), plt.cm.gray)
    plt.show()


def unblur(img):
    negative_laplace = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(img, 5)
    """for i in range(1):
        negative_laplace = signal.convolve2d(negative_laplace, hp_filter, mode='same')"""
    plt.imshow(negative_laplace)
    plt.title("Gaussian")
    plt.show()
    return img - negative_laplace


def main():
    img = cv2.imread('moonlanding.png')
    print(img)
    plt.imshow(img, plt.cm.gray)
    plt.show()
    #grad = signal.convolve2d(img, hp_filter, boundary='symm', mode='same')
    """
    for i in range(5):
        grad = signal.convolve2d(grad, lp_filter, boundary='symm', mode='same')

    """
    new_img = unblur(img)

    plt.imshow(unblur(img), plt.cm.gray)
    plt.show()

if __name__ == "__main__":
    main()
