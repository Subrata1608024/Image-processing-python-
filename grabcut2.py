import cv2 
import numpy as np
from matplotlib import pyplot as plt
import glob

path = "F:/classification_2/test/malignant/*.*"
img_number = 1
for file in glob.glob(path):
      image_bgr = cv2.imread(file)
      image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

      rectangle = (55, 55, 165, 165)

      mask = np.zeros(image_rgb.shape[:2], np.uint8)

      bgdModel = np.zeros((1, 65), np.float64)
      fgdModel = np.zeros((1,65), np.float64)

      cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

      mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

      outputMask = (mask_2 * 255).astype("uint8")

      image_rgd_nobg = image_rgb * mask_2[:, :, np.newaxis]

      imagebg = cv2.cvtColor(image_rgd_nobg, cv2.COLOR_RGB2BGR)
      cv2.imwrite("F:/New folder/grabcut segmentation/test/malignant/mn"+str(img_number)+".jpg",  imagebg)
      img_number +=1

      cv2.waitKey(0)

