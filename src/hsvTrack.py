#!/usr/bin/env python
import cv2
cv2.namedWindow('HSV Calibrator')
import os
import numpy as np
import matplotlib.pyplot as plt

aaa=0
def nothing(a):
    """Does nothing."""
    global aaa
    aaa=1

img = cv2.imread("/home/yuqiang/Pictures/Screenshot from 2021-08-30 17-47-28.png")

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

# h = hsv[0]
# s = hsv[1]
# v = hsv[2]
# img_gray_hist = cv2.calcHist([hsv], [0], None, [180], [0, 179])
# plt.plot(img_gray_hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
# plt.show()
# img_gray_hist = cv2.calcHist([hsv], [1], None, [180], [0, 179])
# plt.plot(img_gray_hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
# plt.show()
# img_gray_hist = cv2.calcHist([hsv], [2], None, [180], [0, 179])
# plt.plot(img_gray_hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
# plt.show()

# while 1:
#     cv2.createTrackbar('H_low', 'HSV Calibrator', 80, 179, nothing)
#     cv2.createTrackbar('S_low', 'HSV Calibrator', 110, 255, nothing)
#     cv2.createTrackbar('V_low', 'HSV Calibrator', 8, 255, nothing) 

#     cv2.createTrackbar('H_high', 'HSV Calibrator', 119, 179, nothing)
#     cv2.createTrackbar('S_high', 'HSV Calibrator', 252, 255, nothing) 
#     cv2.createTrackbar('V_high', 'HSV Calibrator', 122, 255, nothing)

#     h_low = cv2.getTrackbarPos('H_low', 'HSV Calibrator')
#     s_low = cv2.getTrackbarPos('S_low', 'HSV Calibrator')
#     v_low = cv2.getTrackbarPos('V_low', 'HSV Calibrator')
#     h_high = cv2.getTrackbarPos('H_high', 'HSV Calibrator')
#     s_high = cv2.getTrackbarPos('S_high', 'HSV Calibrator')
#     v_high = cv2.getTrackbarPos('V_high', 'HSV Calibrator')
#     lower = np.array([h_low, s_low, v_low])
#     upper = np.array([h_high, s_high, v_high])

#     mask = cv2.inRange(hsv, lower, upper)

#     result = cv2.bitwise_or(img, img, mask=mask)

#     cv2.imshow('HSV Calibrator', result)
#     aaa=0
#     key=cv2.waitKey(0)
#     print(key)
#     if key == 27:
#         break



# b

lower = np.array([85, 60, 8])
upper = np.array([140 , 252, 182])
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_or(img, img, mask=mask)

cv2.imshow('HSV Calibrator', result)
cv2.imshow('HSV', hsv)
key=cv2.waitKey(0)



# g

lower = np.array([67, 40, 30])
upper = np.array([84, 253, 160])
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_or(img, img, mask=mask)

cv2.imshow('HSV Calibrator', result)
cv2.imshow('HSV', hsv)
key=cv2.waitKey(0)

# r

lower = np.array([170, 60, 35])
upper = np.array([179, 255, 180])
mask1 = cv2.inRange(hsv, lower, upper)
lower = np.array([0, 60, 35])
upper = np.array([4, 255, 180])
mask2 = cv2.inRange(hsv, lower, upper)
mask=cv2.bitwise_or(mask1,mask2)

result = cv2.bitwise_or(img, img, mask=mask)

cv2.imshow('HSV Calibrator', result)
cv2.imshow('HSV', hsv)
key=cv2.waitKey(0)