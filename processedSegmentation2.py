import cv2 
import glob
import numpy as np
from matplotlib import pyplot as plt 


path = "F:/New folder/median filter/test/malignant/*.*"
img_number = 1

for file in glob.glob(path):
      img = cv2.imread(file)
      ret, th4 = cv2.threshold(img, 110, 255, cv2.THRESH_TOZERO)
      
      
      cv2.imwrite("F:/New folder/threshold segmentation/test/malignant/mn"+str(img_number)+".jpg", th4  )
      img_number +=1

      cv2.waitKey(0)
      cv2.destroyAllWindows()