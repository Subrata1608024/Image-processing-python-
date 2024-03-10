import numpy as np
import cv2
# construct the argument parser and parse the arguments
image = cv2.imread("C:/Users/User/Desktop/report picture/median filter/malignantg/mg1.jpg")
mask = np.zeros(image.shape[:2], dtype="uint8")

rectangle = (55, 55, 165, 165)
fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")

cv2.grabCut(image, mask, rectangle, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)

for (name, value) in values:
	# construct a mask that for the current value
	print("[INFO] showing mask for '{}'".format(name))
	valueMask = (mask == value).astype("uint8") * 255
    
cv2.imshow(name, valueMask)
cv2.waitKey(0)  

outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
	0, 1) 
outputMask = (outputMask * 255).astype("uint8")

output = cv2.bitwise_and(image, image, mask=outputMask)

cv2.imshow("Input", image)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imshow("GrabCut Output", output)
cv2.waitKey(0)