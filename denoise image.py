# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:34:55 2021

@author: User
"""

import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/classification/HAM10000_images_part_1/ISIC_0024368 (2).jpg" , 1)
kernel = np.ones((3,3), np.float32)/9


filt_2D = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img,(3,3))
gaussian_blur = cv2.GaussianBlur(img, (5,5), 0)
median_blur = cv2.medianBlur(img, 6)


cv2.imshow("Original", img)
cv2.imshow("2D cuctom filter", filt_2D)
cv2.imshow("blur", blur)
cv2.imshow("gaussian_blur", gaussian_blur)
cv2.imshow("median_blur", median_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()