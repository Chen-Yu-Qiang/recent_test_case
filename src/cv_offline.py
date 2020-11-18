#!/usr/bin/env python
import cv2
import os
import numpy as np
def findRect(img):
    def nothing(data):
        pass

    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    print(hsv[360][480])
    # # for g.png
    # lower_g = np.array([30, 0, 0])
    # upper_g = np.array([80, 255, 255])
    # mask=cv2.inRange(hsv, lower_g, upper_g)


    #for r.png
    lower_red = np.array([170, 80, 80])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([0, 80, 80])
    upper_red = np.array([10, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask=cv2.bitwise_or(mask1,mask2)

    # # # for b.png
    # lower_b = np.array([100, 40, 0])
    # upper_b = np.array([120, 255, 255])
    # mask=cv2.inRange(hsv, lower_b, upper_b)


    result = cv2.bitwise_and(img, img, mask=mask)

    kernel = np.ones((7,7), np.uint8)
    erosion = cv2.erode(result, kernel, iterations = 1)

    gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    _,contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    A_max=0
    c_max=None
    for c in contours:
        if cv2.contourArea(c)>A_max:
            A_max=cv2.contourArea(c)
            c_max=c
    
    x, y, w, h = cv2.boundingRect(c_max)
    addbox=cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 5) 
    addbox=cv2.circle(addbox, (480,360), 10, 255)
    cv2.imshow('addbox', addbox)
    cv2.waitKey(0)
    return x,y,w,h


img = cv2.imread(os.getcwd()+"/../rgb.png")
print(findRect(img))

cv2.destroyAllWindows()


