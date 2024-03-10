cv2.imshow("Original", img)
cv2.imshow("2D cuctom filter", filt_2D)
cv2.imshow("blur", blur)
cv2.imshow("gaussian_blur", gaussian_blur)
cv2.imshow("median_blur", median_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
runfile('C:/Users/User/.spyder-py3/untitled3.py', wdir='C:/Users/User/.spyder-py3')

## ---(Wed Jul 14 10:41:24 2021)---
runfile('C:/Users/User/.spyder-py3/untitled3.py', wdir='C:/Users/User/.spyder-py3')

## ---(Wed Jul 14 10:45:15 2021)---
runfile('C:/Users/User/.spyder-py3/untitled3.py', wdir='C:/Users/User/.spyder-py3')
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

## ---(Wed Jul 14 13:02:14 2021)---
runfile('C:/Users/User/.spyder-py3/untitled0.py', wdir='C:/Users/User/.spyder-py3')
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/classification_2/data/test/benign/717.jpg", 0)

plt.hist(img.flat, bins=100, range=(0,255))

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyAllWindows()
2+2
runfile('C:/Users/User/.spyder-py3/untitled0.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled0.py', wdir='C:/Users/User/.spyder-py3')

## ---(Wed Jul 14 13:22:06 2021)---
runfile('C:/Users/User/.spyder-py3/untitled0.py', wdir='C:/Users/User/.spyder-py3')
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/classification_2/data/test/benign/717.jpg", 0)

plt.hist(img.flat, bins=100, range=(0,255))

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/classification_2/data/test/benign/717.jpg", 0)

plt.hist(img.flat, bins=100, range=(0,255))

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## ---(Wed Jul 14 13:42:31 2021)---
runfile('C:/Users/User/.spyder-py3/untitled2.py', wdir='C:/Users/User/.spyder-py3')

## ---(Thu Jul 15 06:29:23 2021)---
runfile('C:/Users/User/.spyder-py3/untitled4.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled5.py', wdir='C:/Users/User/.spyder-py3')

## ---(Thu Jul 15 08:37:59 2021)---
runfile('C:/Users/User/untitled0.py', wdir='C:/Users/User')

## ---(Thu Jul 15 08:49:02 2021)---
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/grabcut.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/save.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/save.py', wdir='C:/Users/User/.spyder-py3')
dir
runfile('C:/Users/User/.spyder-py3/save.py', wdir='C:/Users/User/.spyder-py3')

## ---(Thu Jul 15 10:27:33 2021)---
runfile('C:/Users/User/.spyder-py3/processed1.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/model1.phy.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled4.py', wdir='C:/Users/User/.spyder-py3')

## ---(Thu Jul 15 12:39:27 2021)---
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/model1.phy.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled0.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled1.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/save.py', wdir='C:/Users/User/.spyder-py3')

## ---(Fri Jul 16 05:27:29 2021)---
from keras.preprocessing.image import ImageDataGenerator
from skimage import io



datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_model='constant', cval=25)


x = io.imread('C:/Users/User/Desktop/New folder/thres/th1.jpg')
from keras.preprocessing.image import ImageDataGenerator
from skimage import io



datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant', cval=25)


x = io.imread('C:/Users/User/Desktop/New folder/thres/th1.jpg')
runfile('C:/Users/User/.spyder-py3/untitled2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled1.py', wdir='C:/Users/User/.spyder-py3')

## ---(Fri Jul 16 07:34:38 2021)---
runfile('C:/Users/User/.spyder-py3/medium denoise.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/augmentation.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/augmentation.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/model1.phy.py', wdir='C:/Users/User/.spyder-py3')

## ---(Fri Jul 16 08:49:02 2021)---
runfile('C:/Users/User/.spyder-py3/model1.phy.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled4.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/model1.phy.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled4.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/model1.phy.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')

## ---(Sat Jul 17 06:29:38 2021)---
runfile('C:/Users/User/.spyder-py3/model1.phy.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled4.py', wdir='C:/Users/User/.spyder-py3')

## ---(Sun Jul 18 03:44:51 2021)---
runfile('C:/Users/User/.spyder-py3/fit model.py', wdir='C:/Users/User/.spyder-py3')

## ---(Mon Jul 19 07:30:43 2021)---
runfile('C:/Users/User/.spyder-py3/untitled1.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/medium denoise.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled1.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/fit model.py', wdir='C:/Users/User/.spyder-py3')

## ---(Wed Jul 21 03:03:23 2021)---
runfile('C:/Users/User/.spyder-py3/save.py', wdir='C:/Users/User/.spyder-py3')

## ---(Thu Jul 22 08:16:33 2021)---
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')

## ---(Fri Jul 23 09:26:25 2021)---
runfile('C:/Users/User/.spyder-py3/save.py', wdir='C:/Users/User/.spyder-py3')

## ---(Tue Aug  3 08:53:11 2021)---
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/save.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/untitled4.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')

## ---(Thu Aug  5 06:13:34 2021)---
runfile('C:/Users/User/.spyder-py3/new grabcut1.py', wdir='C:/Users/User/.spyder-py3')

## ---(Mon Aug  9 08:51:45 2021)---
runfile('C:/Users/User/.spyder-py3/autosave/untitled3.py', wdir='C:/Users/User/.spyder-py3/autosave')

## ---(Mon Aug  9 11:20:13 2021)---
runfile('C:/Users/User/.spyder-py3/medium denoise.py', wdir='C:/Users/User/.spyder-py3')

## ---(Mon Aug  9 11:24:09 2021)---
runfile('C:/Users/User/.spyder-py3/medium denoise.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/autosave/untitled3.py', wdir='C:/Users/User/.spyder-py3/autosave')
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/medium denoise.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/save.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/augmentation.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/new grabcut1.py', wdir='C:/Users/User/.spyder-py3')

## ---(Sun Aug 15 10:58:49 2021)---
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/autosave/untitled2.py', wdir='C:/Users/User/.spyder-py3/autosave')
runfile('C:/Users/User/.spyder-py3/autosave/untitled4.py', wdir='C:/Users/User/.spyder-py3/autosave')

## ---(Fri Aug 27 22:53:15 2021)---
runfile('C:/Users/User/.spyder-py3/autosave/untitled4.py', wdir='C:/Users/User/.spyder-py3/autosave')

## ---(Sat Aug 28 08:21:23 2021)---
runfile('C:/Users/User/.spyder-py3/autosave/untitled4.py', wdir='C:/Users/User/.spyder-py3/autosave')

## ---(Sun Aug 29 00:34:06 2021)---
runfile('C:/Users/User/.spyder-py3/autosave/untitled4.py', wdir='C:/Users/User/.spyder-py3/autosave')
runfile('C:/Users/User/.spyder-py3/medium denoise.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/processedSegmentation2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/autosave/untitled4.py', wdir='C:/Users/User/.spyder-py3/autosave')
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')
runfile('C:/Users/User/.spyder-py3/autosave/untitled4.py', wdir='C:/Users/User/.spyder-py3/autosave')

## ---(Mon Aug 30 03:20:40 2021)---
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')

## ---(Sun Sep  5 07:07:04 2021)---
runfile('C:/Users/User/.spyder-py3/grabcut2.py', wdir='C:/Users/User/.spyder-py3')

## ---(Mon Sep  6 03:11:57 2021)---
runfile('C:/Users/User/.spyder-py3/autosave/untitled4.py', wdir='C:/Users/User/.spyder-py3/autosave')