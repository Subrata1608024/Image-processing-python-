import cv2 
import glob
import numpy as np
from matplotlib import pyplot as plt

path = "F:/classification_2/test/malignant/*.*"
img_number = 1

for file in glob.glob(path):
      img = cv2.imread(file)
     
      median_blur = cv2.medianBlur(img, 3)
      
     
      cv2.imwrite("F:/New folder/median filter/test/malignant/mn"+str(img_number)+".jpg",  median_blur )
      img_number +=1
      
      cv2.waitKey(0)
      cv2.destroyAllWindows()