#!/usr/bin/env python
import cv2
import os
import numpy as np

def nothing(a):
    """Does nothing."""
    pass

img = cv2.imread(os.getcwd()+"/../image2.png")

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 



while 1:
    cv2.namedWindow('HSV Calibrator')
    cv2.createTrackbar('H_low', 'HSV Calibrator', 0, 179, nothing)
    cv2.createTrackbar('S_low', 'HSV Calibrator', 0, 255, nothing)
    cv2.createTrackbar('V_low', 'HSV Calibrator', 0, 255, nothing)

    cv2.createTrackbar('H_high', 'HSV Calibrator', 50, 179, nothing)
    cv2.createTrackbar('S_high', 'HSV Calibrator', 100, 255, nothing)
    cv2.createTrackbar('V_high', 'HSV Calibrator', 100, 255, nothing)

    h_low = cv2.getTrackbarPos('H_low', 'HSV Calibrator')
    s_low = cv2.getTrackbarPos('S_low', 'HSV Calibrator')
    v_low = cv2.getTrackbarPos('V_low', 'HSV Calibrator')
    h_high = cv2.getTrackbarPos('H_high', 'HSV Calibrator')
    s_high = cv2.getTrackbarPos('S_high', 'HSV Calibrator')
    v_high = cv2.getTrackbarPos('V_high', 'HSV Calibrator')
    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])

    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('HSV Calibrator', result)
    key=cv2.waitKey(1000)
    print(key)
    if key == 27:
        break