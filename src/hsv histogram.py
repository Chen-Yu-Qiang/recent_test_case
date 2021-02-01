import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
print(os.getcwd())
img = cv2.imread('./time_RGB/g_20.png')

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h = hsv[0]
s = hsv[1]
v = hsv[2]
img_gray_hist = cv2.calcHist([hsv], [0], None, [180], [0, 179])
plt.plot(img_gray_hist)
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.show()
img_gray_hist = cv2.calcHist([hsv], [1], None, [180], [0, 179])
plt.plot(img_gray_hist)
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.show()
img_gray_hist = cv2.calcHist([hsv], [2], None, [180], [0, 179])
plt.plot(img_gray_hist)
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.show()