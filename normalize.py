# 1. Normalize
# 2. Segmentation
# 3. Coherence Diffusion Filter
# 4. Log-Gabor Filter
# 5. Binarization

import cv2
import numpy as np
  

img = cv2.imread('test.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

_ , thresh = cv2.threshold(normalized, np.mean(normalized), 255, cv2.THRESH_BINARY)

imgResized = cv2.resize(thresh, (960, 540))

cv2.imshow('Threshold', imgResized)
cv2.waitKey(0)