import cv2 
import glob
import numpy as np
from matplotlib import pyplot as plt 


path = "C:/classification_2/data/test/malignant/*.*"
img_number = 1



for file in glob.glob(path):
    image_bgr = cv2.imread(file)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    rectangle = (70, 70, 100, 100)

    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    image_rgd_nobg = image_rgb * mask_2[:, :, np.newaxis]
    
    plt.imshow(image_rgd_nobg)
    plt.show()
  
    
    

    cv2.imwrite("C:/processed image/malignant1/ma"+str(img_number)+".jpg", image_rgd_nobg  )
    img_number +=1

    
    
