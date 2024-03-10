import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/classification_2/data/test/benign/717.jpg", 0)

plt.hist(img.flat, bins=100, range=(0,255))

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
