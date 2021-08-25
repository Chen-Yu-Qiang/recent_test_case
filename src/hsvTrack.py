#!/usr/bin/env python
import cv2
cv2.namedWindow('HSV Calibrator')
import os
import numpy as np


aaa=0
def nothing(a):
    """Does nothing."""
    global aaa
    aaa=1

img = cv2.imread("/home/yuqiang/Pictures/Screenshot from 2021-08-25 14-43-32.png")

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 



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




lower = np.array([80, 50, 8])
upper = np.array([119, 252, 182])
mask = cv2.inRange(hsv, lower, upper)

result = cv2.bitwise_or(img, img, mask=mask)

cv2.imshow('HSV Calibrator', result)
cv2.imshow('HSV', hsv)
key=cv2.waitKey(0)