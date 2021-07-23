#!/usr/bin/env python
import cv2
import os
import numpy as np
import time
from cv2 import aruco

def find_aruco_mean(img):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict)
    # print(corners, ids)
    if ids is None:
        return -1,-1
    x=[0 for i in range(len(corners))]
    y=[0 for i in range(len(corners))]
    idss=[0 for i in range(len(corners))]
    xyid=[0 for i in range(len(corners))]
    max_x=[0 for i in range(len(corners))]
    for i in range(len(corners)):
        for j in range(4):
            x[i]=x[i]+corners[i][0][j][0]/4
            y[i]=y[i]+corners[i][0][j][1]/4
            max_x[i]=max(max_x[i],corners[i][0][j][0])
        idss[i]=ids[i][0]

        print(xyid)
        xyid[i]=[x[i],y[i],idss[i]]
    # print(xyid)
    if len(xyid)>=1:
        ip=[0 for i in range(len(x))]
        xyid=sorted(xyid, key=lambda s: s[0])
        xyid=xyid+[[960,360,-1]]
        for i in range(len(xyid)-1):
            ip[i]=(xyid[i+1][0]-xyid[i][0])/4+xyid[i][0]
        if i==(len(xyid)-2) and (ip[i]<max(max_x)):
            ip[i]=xyid[i][0]+(max(max_x)-xyid[i][0])*2
            ip[i]=max(ip[i],960)
        xyid=xyid[0:-1]
    return xyid,ip

def divImg(ip,img):
    img_set=[img for i in range(len(ip))]
    if len(ip)>=1:
        ip=[0]+ip
        for i in range(len(ip)-1):
            sss=np.zeros([720,960],dtype=np.uint8)
            sss[0:720,int(ip[i]):int(ip[i+1])]=255
            img_set[i]=cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=sss)

    return img_set
if __name__ == '__main__':
    filename="/home/yuqiang/catkin_ws4/src/recent_test_case/2target.png"
    img = cv2.imread(filename)
    cv2.imshow("q",img)
    xyid,ip = find_aruco_mean(img)
    print(xyid,ip)
    img_set=divImg(ip,img)
    for i in img_set:
        cv2.imshow("q2",i)
        cv2.waitKey(0)