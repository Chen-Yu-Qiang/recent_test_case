import cv2
import numpy as np


img = cv2.imread('r2m.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 


lower_red = np.array([170, 80, 80])
upper_red = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)
lower_red = np.array([0, 80, 80])
upper_red = np.array([10, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)
mask=cv2.bitwise_or(mask1,mask2)

result = cv2.bitwise_and(img, img, mask=mask)

kernel = np.ones((27,27), np.uint8)
erosion = cv2.erode(result, kernel, iterations = 1)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
cv2.imshow('image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows
#ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print(contours)
for i in contours:
    img = cv2.imread('r2m.png')
    size = cv2.contourArea(i)
    rect = cv2.minAreaRect(i)
    if size >10000:
        gray = np.float32(gray)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.fillPoly(mask, [i], (255,255,255))
        dst = cv2.cornerHarris(mask,5,3,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        if rect[2] == 0 and len(corners) == 5:
            x,y,w,h = cv2.boundingRect(i)
            if w == h or w == h +3: #Just for the sake of example
                print('Square corners: ')
                for i in range(1, len(corners)):
                    print(corners[i])
            else:
                print('Rectangle corners: ')
                for i in range(1, len(corners)):
                    print(corners[i])
        if len(corners) == 5 and rect[2] != 0:
            print('Rombus corners: ')
            for i in range(1, len(corners)):
                print(corners[i])
        if len(corners) == 4:
            print('Triangle corners: ')
            for i in range(1, len(corners)):
                print(corners[i])
        if len(corners) == 6:
            print('Pentagon corners: ')
            for i in range(1, len(corners)):
                print(corners[i])
        img[dst>0.1*dst.max()]=[0,0,255]
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows