from skimage import io

from matplotlib import pyplot as plt

img = io.imread("C:/classification_2/EEe/apd.jpg", as_gray=True)

from skimage import restoration

import numpy as np
psf = np.ones((3,3)) /9

deconvoled,_ =restoration.unsupervised_wiener(img, psf)

plt.imsave("C:/classification_2/EEe/deconvoled.jpg", deconvoled, cmap ="gray")


