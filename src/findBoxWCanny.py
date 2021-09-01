#!/usr/bin/env python
import cv2
import os
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import pnp
from cv2 import aruco
r=None
g=None
b=None
DEBUG_MODE=0

HSVrang={
    "rL1":[174, 165, 35],
    "rH1":[179, 255, 164],
    "rL2":[0, 165, 35],
    "rH2":[4, 255, 164],
    "gL":[67,0,0],
    "gH":[78,255,255],
    "bL":[80,0,0],
    "bH":[120,255,255],
}
HSVrang={
    "rL1":[171, 165, 111],
    "rH1":[179, 255, 211],
    "rL2":[0, 165, 111],
    "rH2":[10, 255, 211],
    "gL":[67, 102, 109],
    "gH":[78, 229, 251],
    "bL":[103, 126, 100],
    "bH":[116, 210, 180]
}
def findRect(img,color):
    global HSVrang
    tm_hour=10
    # convert to HSV
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    if color=="g":
        if tm_hour==25:
            lower_g = np.array(HSVrang["gL"])
            upper_g = np.array(HSVrang["gH"])

        elif tm_hour<=18 and tm_hour>=6 :
            lower_g = np.array([67, 70, 30])
            upper_g = np.array([78, 253, 160])
        else:
            lower_g = np.array([67, 102, 109])
            upper_g = np.array([78, 229, 251])

        mask=cv2.inRange(hsv, lower_g, upper_g)

 
    if color=="r":

        if tm_hour==25:
            lower_red = np.array(HSVrang["rL1"])
            upper_red = np.array(HSVrang["rH1"])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array(HSVrang["rL2"])
            upper_red = np.array(HSVrang["rH2"])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)

        elif tm_hour<=18 and tm_hour>=6 :
            lower_red = np.array([174, 60, 35])
            upper_red = np.array([179, 255, 164])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 60, 35])
            upper_red = np.array([4, 255, 164])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
        else:
            lower_red = np.array([171, 165, 111])
            upper_red = np.array([179, 255, 211])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 165, 111])
            upper_red = np.array([10, 255, 211])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
        mask=cv2.bitwise_or(mask1,mask2)
        # cv2.imshow('mask'+str(color), mask)
        # cv2.waitKey(1)

    if color=="b":

        if tm_hour==25:
            lower_b = np.array(HSVrang["bL"])
            upper_b = np.array(HSVrang["bH"])

        elif tm_hour<=18 and tm_hour>=6 :
            lower_b = np.array([85, 60, 8])
            upper_b = np.array([140, 252, 122])
        else:
            lower_b = np.array([108, 126, 124])
            upper_b = np.array([116, 175, 161])
            lower_b = np.array([103, 126, 100])
            upper_b = np.array([116, 210, 180])
        mask=cv2.inRange(hsv, lower_b, upper_b)


    
    # cv2.imshow('mask'+str(color), mask)
    # cv2.waitKey(1)
    # mask = cv2.dilate(mask, np.ones((17,17), np.uint8), iterations = 1)
    result = cv2.bitwise_and(img, img, mask=mask)

    # new
    # kernel = np.ones((17,17), np.uint8)
    # result = cv2.dilate(result, kernel, iterations = 1)

    # # old
    # kernel = np.ones((27,27), np.uint8)
    # result = cv2.erode(result, kernel, iterations = 1)


    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if DEBUG_MODE:
        cv2.imshow('gray'+str(color), gray)
        cv2.waitKey(0)
    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if DEBUG_MODE:
        cv2.imshow('threshold'+str(color), binary)
        cv2.waitKey(0)


    # cv2.imshow('erosion'+str(color), erosion)
    # cv2.waitKey(0)


    kernel = np.ones((5,5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations = 1)
    

    _,contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # img2=cv2.drawContours(img,contours,-1,(0,255,0),5)
    # cv2.imshow('erosion2'+str(color), img2)
    # cv2.waitKey(1)
    # print(len(contours),color)
    A_max=0
    c_max=None
    []
    for c in contours: 
        __, _, w1, h1 = cv2.boundingRect(c)
        # print(cv2.boundingRect(c))
        if cv2.contourArea(c)>A_max and w1>50 and h1>70 :
            A_max=cv2.contourArea(c)
            c_max=c

    # peri = cv2.arcLength(c_max, True) 
    # approx1 = cv2.approxPolyDP(c_max, 0.05*peri, True)
    # # print(len(approx1),len(approx1[0]))
    # img2=cv2.polylines(img, [approx1], True, (255, 0, 0), 1) 
    # if DEBUG_MODE:
    #     cv2.imshow('img2', img2)
    #     cv2.waitKey(0)
    # print(approx1)

    x, y, w, h = cv2.boundingRect(c_max)
    if DEBUG_MODE==1:
        img=cv2.drawContours(img,contours,-1,(0,255,0),5)
        addbox=cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1) 
        print(x,y,w,h,color)
        cv2.imshow("ad",addbox)
        cv2.waitKey(1)
   
    return x,y,w,h

def findCanny(img,color):
    # global r,g,b
    if color=="r":
        r=None
    if color=="g":
        g=None
    if color=="b":
        b=None

    bigger=0.2
    # cv2.imshow("org"+str(color),cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # cv2.waitKey(1)
    img	=cv2.undistort(img, np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]]), np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000]))

    x,y,w,h = findRect(img,color)

    simp=xywh2approx1(x,y,w,h)
    if w*h==0:
        print("findRect=0",color)
        return 
    mask = np.zeros((720,960,1), np.uint8) 
    mask.fill(0)
    cv2.rectangle(mask, (int(x-w*bigger), int(y-h*bigger)), (int(x+w*(1+bigger)), int(y+h*(1+bigger))), 255, -1)


    image = cv2.bitwise_or(img, img, mask=mask)

    #image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    y_min=max(0,int(y-1.0*h*bigger+1))
    y_max=min(719,int(y+h*(1+bigger)-1))
    x_min=max(0,int(x-1.0*w*bigger+1))
    x_max=min(959,int(x+1.0*w*(1.0+bigger)-1))
    # cv2.imshow('cut'+color, image[y_min:y_max,x_min:x_max])
    # cv2.waitKey(1)   
    # print(image.shape)
    blurred = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('blurred', blurred)
    blurred2=blurred[y_min:y_max,x_min:x_max]

    # print("G",canny.shape)
    blurred2=255-blurred2
    if DEBUG_MODE:
        cv2.imshow('canny'+color, blurred2)
        cv2.waitKey(0)


    ret,binary = cv2.threshold(blurred2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if DEBUG_MODE:
        cv2.imshow('binary2'+color, binary)
        cv2.waitKey(0)


    # canny=================================
    # canny = cv2.Canny(canny, 70, 120) 
    # if DEBUG_MODE==1:
    #     cv2.imshow('canny2'+color, canny)
    #     cv2.waitKey(1)


    # if canny is None:
    #     print("canny is none",color)
    #     return simp
    # canny = cv2.GaussianBlur(canny, (3, 3), 0)
    # # cv2.imshow('canny'+color, canny)
    # # cv2.waitKey(0)
    # binary=canny

    # canny=================================


    # print("F")
    _,contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for ii in range(len(contours)):
        for i in range(len(contours[ii])):
            # print(contours[ii][i][0][0],contours[ii][i][0][1])
            contours[ii][i]=[[contours[ii][i][0][0]+x_min,contours[ii][i][0][1]+y_min]] 

    # cv2.drawContours(image,contours,-1,(0,0,255),2) 
    # if DEBUG_MODE==1:
    #     cv2.imshow('image'+color, image)
    #     cv2.waitKey(1)
    A_max=0
    c_max=None
    # print(len(contours))
    for c in contours:
        __, _, w1, h1 = cv2.boundingRect(c) 
        # print(cv2.boundingRect(c))
        if cv2.contourArea(c)>A_max and w1>50 and h1>70 :
            A_max=cv2.contourArea(c)
            c_max=c
    if A_max==0:
        print("no max area",color)
        return simp
    #print(c_max)
    # cv2.drawContours(image,c_max,-1,(0,0,255),2) 
    # cv2.imshow('mavimage'+color, image)
    # cv2.waitKey(1)
    peri = cv2.arcLength(c_max, True) 
    approx1 = cv2.approxPolyDP(c_max, 0.05*peri, True)
    # print(len(approx1),len(approx1[0]))
    cv2.polylines(image, [approx1], True, (255, 0, 0), 1) 
    if DEBUG_MODE:
        cv2.imshow('imageHSV', image)
        cv2.waitKey(0)
        # print(approx1)
    if  not len(approx1)==4:
        # print("len(approx1)=",len(approx1),color)
        # print(approx1)
        if len(approx1)<4:
            return simp
        else:
            approx1=findnear(approx1)
            # print(approx1)
    # print(approx1)
    # print("=== "+str(color)+" good===")
    # H_max,S_max,V_max,H_std,S_std,V_std = gethsv(img,approx1)
    # print(color,H_max,S_max,V_max,H_std,S_std,V_std)
    # nowHSV(color,H_max,S_max,V_max,H_std,S_std,V_std)
    # print("")
    if color=="r":
        r=approx1
    elif color=="g":
        g=approx1
    elif color=="b":
        b=approx1
    # cv2.waitKey(1)
    # findRect(img,color)
    return approx1

def div1234(pp):
    p=[[[0,0,0]],[[0,0,0]],[[0,0,0]],[[0,0,0]]]
    # print(pp)
    for i in range(4):
        p[i][0]=[pp[i][0][0],pp[i][0][1],0]

    p.sort(key=(lambda x:x[0][0]))
    p[2][0][2]=1
    p[3][0][2]=1
    p.sort(key=(lambda x:x[0][1]))
    p[2][0][2]=p[2][0][2]+2
    p[3][0][2]=p[3][0][2]+2
    p.sort(key=(lambda x:x[0][2]))
    x3=p[1][0][0]
    x4=p[2][0][0]
    x2=p[3][0][0]
    y1=p[0][0][1]
    y3=p[1][0][1]
    x1=p[0][0][0]
    y4=p[2][0][1]
    y2=p[3][0][1]

    return [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

def div1234_plotPoly(pp):
    if pp is None:
        return None
    p=[[[0,0,0]],[[0,0,0]],[[0,0,0]],[[0,0,0]]]
    # print(pp)
    for i in range(4):
        p[i][0]=[pp[i][0][0],pp[i][0][1],0]

    p.sort(key=(lambda x:x[0][0]))
    p[2][0][2]=1
    p[3][0][2]=1
    p.sort(key=(lambda x:x[0][1]))
    p[2][0][2]=p[2][0][2]+2
    p[3][0][2]=p[3][0][2]+2
    p.sort(key=(lambda x:x[0][2]))
    x3=p[1][0][0]
    x4=p[2][0][0]
    x2=p[3][0][0]
    y1=p[0][0][1]
    y3=p[1][0][1]
    x1=p[0][0][0]
    y4=p[2][0][1]
    y2=p[3][0][1]

    return np.array([[[x1,y1]],[[x3,y3]],[[x2,y2]],[[x4,y4]]])

def xywh2approx1(x,y,w,h):
    p1=[x,y]
    p2=[x+w,y+h]
    p3=[x+w,y]
    p4=[x,y+h]
    return np.array([[p1],[p2],[p3],[p4]])

def xywh(p):
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]=p
    x=(x1*x3*y2 - x2*x3*y1 - x1*x4*y2 + x2*x4*y1 - x1*x3*y4 + x1*x4*y3 + x2*x3*y4 - x2*x4*y3)/(x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)
    y=(x1*y2*y3 - x2*y1*y3 - x1*y2*y4 + x2*y1*y4 - x3*y1*y4 + x4*y1*y3 + x3*y2*y4 - x4*y2*y3)/(x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)
    # (x1,y1)  (x3,y3)
    # (x4,y4)  (x2,y2)
    w=0.5*(1.0*x3+x2-x1-x4)
    h=0.5*(1.0*y2+y4-y1-y3)
    a=-0.5*(x2*y3+x3*y1+x1*y4+x4*y2-x3*y2-x1*y3-x4*y1-x2*y4)
    # print(p,w,h,a)
    # print(1.0*x-(x1+x2+x3+x4)*0.25)
    # print(1.0*y-(y1+y2+y3+y4)*0.25)
    return x,y,w,h,a

def mySTD(a,m,l):
    b=0
    c=0
    for i in range(1,l):
        b=b+(i-m)*(i-m)*a[i]
        c=c+a[i]
    return np.sqrt(b/c)

def gethsv(img,p):
    mask = np.zeros((720,960,1), np.uint8) 
    mask.fill(0)
    cv2.fillPoly(mask,[p],(255,255,255))

    image = cv2.bitwise_or(img, img, mask=mask)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv123",image)
    # cv2.waitKey(1)
    img_gray_hist = cv2.calcHist([hsv], [0], None, [180], [0, 179])
    # plt.plot(img_gray_hist[1:])
    # plt.show()
    H_max=img_gray_hist.tolist().index(max(img_gray_hist[1:]))
    if H_max<10 or H_max>170:
        if H_max<90:
            H_max_shift=H_max+89
        else:
            H_max_shift=H_max-90
        H_0_mean=(img_gray_hist[1]+img_gray_hist[179])/2
        # print(H_0_mean)
        H_shift=np.append(img_gray_hist[90:180],H_0_mean)
        H_shift=np.append(H_shift,img_gray_hist[1:90])
        H_shift=np.append(0,H_shift)
        H_std=mySTD(H_shift,H_max_shift,181)
        # plt.plot(H_shift)
        # plt.show()    
    else:
        # print(H_max)
        H_std=mySTD(img_gray_hist,H_max,180)
    img_gray_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    # plt.plot(img_gray_hist[1:-1])
    # plt.show()
    S_max=img_gray_hist.tolist().index(max(img_gray_hist[1:]))
    S_std=mySTD(img_gray_hist,S_max,256)
    img_gray_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    # plt.plot(img_gray_hist[1:])
    # plt.show()
    V_max=img_gray_hist.tolist().index(max(img_gray_hist[1:]))
    V_std=mySTD(img_gray_hist,V_max,256)
    # print("STD",H_std,S_std,V_std)
    return H_max,S_max,V_max,H_std,S_std,V_std

def nowHSV(color,H_max,S_max,V_max,H_std,S_std,V_std):
    global HSVrang
    # print(HSVrang)
    manyTime_H=3
    manyTime_S=3
    manyTime_V=3
    NowRate=0.05
    if color=="r":
        S_g_low = S_max-manyTime_S*S_std
        S_g_High = S_max+manyTime_S*S_std
        V_g_low = V_max-manyTime_V*V_std
        V_g_High = V_max+manyTime_V*V_std
        HSVrang["rL1"][1]=int(HSVrang["rL1"][1]*(1-NowRate)+S_g_low*NowRate)
        HSVrang["rL1"][2]=int(HSVrang["rL1"][2]*(1-NowRate)+V_g_low*NowRate)
        HSVrang["rH1"][1]=int(HSVrang["rH1"][1]*(1-NowRate)+S_g_High*NowRate)
        HSVrang["rH1"][2]=int(HSVrang["rH1"][2]*(1-NowRate)+V_g_High*NowRate)
        HSVrang["rL2"][1]=int(HSVrang["rL2"][1]*(1-NowRate)+S_g_low*NowRate)
        HSVrang["rL2"][2]=int(HSVrang["rL2"][2]*(1-NowRate)+V_g_low*NowRate)
        HSVrang["rH2"][1]=int(HSVrang["rH2"][1]*(1-NowRate)+S_g_High*NowRate)
        HSVrang["rH2"][2]=int(HSVrang["rH2"][2]*(1-NowRate)+V_g_High*NowRate)
        if H_max<90:
            H_g_low = H_max-manyTime_H*H_std+180
            H_g_High = H_max+manyTime_H*H_std
        else:
            H_g_low = H_max-manyTime_H*H_std
            H_g_High = H_max+manyTime_H*H_std-180
        HSVrang["rL1"][0]=int(HSVrang["rL1"][0]*(1-NowRate)+H_g_low*NowRate)
        HSVrang["rH2"][0]=int(HSVrang["rH2"][0]*(1-NowRate)+H_g_High*NowRate)
    
    elif color=="g":
        H_g_low = H_max-manyTime_H*H_std
        H_g_High = H_max+manyTime_H*H_std
        S_g_low = S_max-manyTime_S*S_std
        S_g_High = S_max+manyTime_S*S_std
        V_g_low = V_max-manyTime_V*V_std
        V_g_High = V_max+manyTime_V*V_std
        HSVrang["gL"][0]=int(HSVrang["gL"][0]*(1-NowRate)+H_g_low*NowRate)
        HSVrang["gL"][1]=int(HSVrang["gL"][1]*(1-NowRate)+S_g_low*NowRate)
        HSVrang["gL"][2]=int(HSVrang["gL"][2]*(1-NowRate)+V_g_low*NowRate)
        HSVrang["gH"][0]=int(HSVrang["gH"][0]*(1-NowRate)+H_g_High*NowRate)
        HSVrang["gH"][1]=int(HSVrang["gH"][1]*(1-NowRate)+S_g_High*NowRate)
        HSVrang["gH"][2]=int(HSVrang["gH"][2]*(1-NowRate)+V_g_High*NowRate)
    elif color=="b":
        H_g_low = H_max-manyTime_H*H_std
        H_g_High = H_max+manyTime_H*H_std
        S_g_low = S_max-manyTime_S*S_std
        S_g_High = S_max+manyTime_S*S_std
        V_g_low = V_max-manyTime_V*V_std
        V_g_High = V_max+manyTime_V*V_std
        HSVrang["bL"][0]=int(HSVrang["bL"][0]*(1-NowRate)+H_g_low*NowRate)
        HSVrang["bL"][1]=int(HSVrang["bL"][1]*(1-NowRate)+S_g_low*NowRate)
        HSVrang["bL"][2]=int(HSVrang["bL"][2]*(1-NowRate)+V_g_low*NowRate)
        HSVrang["bH"][0]=int(HSVrang["bH"][0]*(1-NowRate)+H_g_High*NowRate)
        HSVrang["bH"][1]=int(HSVrang["bH"][1]*(1-NowRate)+S_g_High*NowRate)
        HSVrang["bH"][2]=int(HSVrang["bH"][2]*(1-NowRate)+V_g_High*NowRate)

    print(HSVrang)

def findnear(d):
    def dis(x1,x2):
        a=np.sqrt((x1[0][0]-x2[0][0])*(x1[0][0]-x2[0][0])+(x1[0][1]-x2[0][1])*(x1[0][1]-x2[0][1]))
        return a

    res=[[[0,0]],[[0,0]],[[0,0]],[[0,0]]]
    now_i=0
    for i in range(len(d)): 
        now_j=1
        for j in range(now_i):
            if dis(d[i],res[j])<20:
                now_j=0
        if now_j==1:
            res[now_i]=d[i].tolist()
            now_i=now_i+1
            if now_i==4:
                break
    return np.array(res)

def rotationMatrixToEulerAngles(R) :

    sy = np.sqrt(R[0,0] * R[0,0] +  R[0,1] * R[0,1])
    
    singular = sy < 1e-6
    if  not singular :
        x = np.arctan2(R[0,1] , R[0,0])
        y = np.arctan2(-R[0,2], sy)
        z = np.arctan2(R[1,2], R[2,2])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x*57.296, y*57.296, z*57.296])

def vp2ang(ph,pv):
    ph=np.array([[ph[0]],[ph[1]],[1]])
    pv=np.array([[pv[0]],[pv[1]],[1]])
    k=np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
    k_inv=np.linalg.inv(k)
    k_inv_ph=np.dot(k_inv,ph)
    if ph[0]<0:
        r2=-(k_inv_ph)/(np.linalg.norm(k_inv_ph))
    else:
        r2=(k_inv_ph)/(np.linalg.norm(k_inv_ph))
    k_inv_pv=np.dot(k_inv,pv)
    if pv[0]<0:
        r3=(k_inv_pv)/(np.linalg.norm(k_inv_pv))
    else:
        r3=-(k_inv_pv)/(np.linalg.norm(k_inv_pv))
    r1=np.cross(r2.reshape((1,3))[0],r3.reshape((1,3))[0])
    R=np.array([[r1[0],r2[0][0],r3[0][0]],[r1[1],r2[1][0],r3[1][0]],[r1[2],r2[2][0],r3[2][0]]])
    # print(r2)
    return np.arctan2(r2[0],r2[2])[0]

def findCanny_mp(i):
    return findCanny(i[0],i[1])
pool=ThreadPool()
def findRGB(img):

    r=None
    g=None
    b=None
    t=time.time()
    
    
    img	=cv2.undistort(img, np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]]), np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000]))
    
    
    # ===========================Multithreading
    # r_jod=threading.Thread(target = findCanny, args = (img,"r"))
    # g_jod=threading.Thread(target = findCanny, args = (img,"g"))
    # b_jod=threading.Thread(target = findCanny, args = (img,"b"))
    # r_jod.start()
    # # # print("r start")
    # g_jod.start()
    # # # print("g start")
    # b_jod.start()
    # # # print("b start")
    # r_jod.join()
    # # # print("r join")
    # g_jod.join()
    # # # print("g join")
    # b_jod.join()
    # # # print("b join")
    # # # print(xywh(div1234(r))) 

    # ===========================Multithreading
    if not DEBUG_MODE:
        res = pool.map(findCanny_mp, [(img,"r"),(img,"g"),(img,"b")])
        r=res[0]
        g=res[1]
        b=res[2]


    # ==========================Single thread
    if DEBUG_MODE:
        r=findCanny(img,"r")
        g=findCanny(img,"g")
        b=findCanny(img,"b")

    eee=0
    if not r is None:
        div1234_point=div1234(r)
        x,y,w,h,a= xywh(div1234_point)
        img=cv2.circle(img, (x,y), 2, 255)
        if x<100 or x>860:
            eee=1
        if not isConvex(r):
            eee=1
            r=None
            print("r not isConvex")
        # print("r",x,y,w,h,a)
        # print(pnp.mypnp(div1234_point))
    if not g is None:
        div1234_point=div1234(g)
        x,y,w,h,a= xywh(div1234_point)
        img=cv2.circle(img, (x,y), 2, 255)
        if x<100 or x>860:
            eee=1
        if not isConvex(r):
            eee=1
            g=None
            print("g not isConvex")
        # print("g",x,y,w,h,a)
        # print(pnp.mypnp(div1234_point))
    if not b is None:
        div1234_point=div1234(b)
        x,y,w,h,a= xywh(div1234_point)
        img=cv2.circle(img, (x,y), 2, 255)
        if x<100 or x>860:
            pass
        if not isConvex(b):
            b=None
            print("b not isConvex")

        # print("b",x,y,w,h,a)
        # print(pnp.mypnp(div1234_point))

    cv2.polylines(img, [div1234_plotPoly(r),div1234_plotPoly(g),div1234_plotPoly(b)], True, ( 0,255, 0), 1)
    # print(time.time()-t)
    if DEBUG_MODE==1:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        cv2.imshow("a",hsv)
    ang=None
    if (not r is None )and (not g is None) and eee==0:
        # p1234To3D(img)
        _,m1,c1,m2,c2,infp_xh,infp_yh=twoLineAngH(div1234(r),div1234(g))
        if twoLineAngH_error(div1234(r),div1234(g)) <10:
            # print(infp_xh,infp_yh,m1,c1,m2,c2)
            img=cv2.line(img,(0,int(c1)),(960,int(960*m1+c1)),(255, 0, 0), 1)
            img=cv2.line(img,(0,int(c2)),(960,int(960*m2+c2)),(255, 0, 0), 1)
            ang=vp2ang((infp_xh,infp_yh),(infp_xh,infp_yh))
            # print("h1",ang)
        else:
            print("LS error too big")
    # cv2.imshow("a",img)
    # cv2.waitKey(1)

    return r,g,b,ang


def d1d2(x,y):
    t=np.pi/2-np.arccos((0.16+x*x-y*y)/(0.8*x))
    d=x*np.cos(t)
    return t/np.pi*180+90,d

def p1234To3D(img=None):
    global r,g,b
    cameraMatrix=np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
    distCoeffs=np.array([0.000000, 0.000000, 0.000000, 0.000000, 0.000000])
    if not r is None:
        div1234_point=div1234(r)
        r_xyz,_,r_t,r_r=pnp.mypnp(div1234_point)
    if not g is None:
        div1234_point=div1234(g)
        g_xyz,_,g_t,g_r=pnp.mypnp(div1234_point)
    if not b is None:
        div1234_point=div1234(b)
        b_xyz,_,b_t,b_r=pnp.mypnp(div1234_point)
    t_tc,t_rc,t_t,t_r=pnp.mypnpTotal(div1234(r),div1234(g),div1234(b))

    t_r, _ = cv2.Rodrigues(t_r)
    if not img is None:
        img=aruco.drawAxis(img, cameraMatrix, distCoeffs,t_r, t_t,0.3)
        # cv2.imshow("Axes",img)
        # cv2.waitKey(1)

    return t_tc,t_rc

def twoLineAngH(pr,pg):
    # (x1,y1)  (x3,y3)
    # (x4,y4)  (x2,y2)
    [[x11,y11],[x12,y12],[x13,y13],[x14,y14]]=pr
    [[x21,y21],[x22,y22],[x23,y23],[x24,y24]]=pg
    A=np.array([[x13,1],[x11,1],[x23,1],[x21,1]])
    y=np.array([y13,y11,y23,y21])
    a,b=np.linalg.lstsq(A, y, rcond=None)[0]
    A=np.array([[x12,1],[x14,1],[x22,1],[x24,1]])
    y=np.array([y12,y14,y22,y24])
    c,d=np.linalg.lstsq(A, y, rcond=None)[0]
    delta=np.arctan(a)-np.arctan(c)
    x=-(b - d)/(a - c)
    y=(a*d - c*b)/(a - c)
    # a b : line1 y=ax+b
    # c d : line2 y=cx+d
    # x,y : Vanishing point
    # delta : angle between 2 line
    return delta,a,b,c,d,x,y

def twoLineAngH_error(pr,pg):
    # (x1,y1)  (x3,y3)
    # (x4,y4)  (x2,y2)
    [[x11,y11],[x12,y12],[x13,y13],[x14,y14]]=pr
    [[x21,y21],[x22,y22],[x23,y23],[x24,y24]]=pg


    A=np.array([[x13,1],[x11,1],[x23,1],[x21,1]])
    y=np.array([y13,y11,y23,y21])
    e1=np.linalg.lstsq(A, y, rcond=None)[1]


    A=np.array([[x12,1],[x14,1],[x22,1],[x24,1]])
    y=np.array([y12,y14,y22,y24])
    e2=np.linalg.lstsq(A, y, rcond=None)[1]

    return max(e1,e2)

def twoLineAngH2(pr,pg):
    # (x1,y1)  (x3,y3)
    # (x4,y4)  (x2,y2)
    [[x11,y11],[x12,y12],[x13,y13],[x14,y14]]=pr
    [[x21,y21],[x22,y22],[x23,y23],[x24,y24]]=pg
    A=np.array([[(y13-y11),(x11-x13)],[(y13-y21),(x21-x13)],[(y23-y21),(x21-x23)],[(y12-y14),(x14-x12)],[(y12-y24),(x24-x12)],[(y22-y24),(x24-x22)]])
    y=np.array([(x11*y13-x13*y11),(x21*y13-x13*y21),(x21*y23-x23*y21),(x14*y12-x12*y14),(x24*y12-x12*y24),(x24*y22-x22*y24)])
    a,b=np.linalg.lstsq(A, y, rcond=None)[0]

    return a,b
def twoLineAngV(pg,pb):
    # (x1,y1)  (x3,y3)
    # (x4,y4)  (x2,y2)
    [[x11,y11],[x12,y12],[x13,y13],[x14,y14]]=pg
    [[x21,y21],[x22,y22],[x23,y23],[x24,y24]]=pb
    A=np.array([[y14,1],[y11,1],[y24,1],[y21,1]])
    y=np.array([x14,x11,x24,x21])
    a,b=np.linalg.lstsq(A, y, rcond=None)[0]
    A=np.array([[y12,1],[y13,1],[y22,1],[y23,1]])
    y=np.array([x12,x13,x22,x23])
    c,d=np.linalg.lstsq(A, y, rcond=None)[0]
    delta=np.arctan(a)-np.arctan(c)
    y=-(b - d)/(a - c)
    x=(a*d - c*b)/(a - c)
    return delta,a,b,c,d,x,y


def isConvex(p):
    if p is None:
        return 0
    def ang123(x11,y11,x22,y22,x33,y33):
        v1x=x11-x22
        v1y=y11-y22
        v2x=x33-x22
        v2y=y33-y22
        l1=np.sqrt(v1x*v1x+v1y*v1y)
        l2=np.sqrt(v2x*v2x+v2y*v2y)
        innPro=v1x*v2x+v1y+v2y
        ang=np.arccos(1.0*innPro/(l1*l2))
        return ang

    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]=div1234(p)
    ang1=ang123(x4,y4,x1,y1,x3,y3)
    ang2=ang123(x1,y1,x3,y3,x2,y2)
    ang3=ang123(x3,y3,x2,y2,x4,y4)
    ang4=ang123(x2,y2,x4,y4,x1,y1)
    totang=ang1+ang2+ang3+ang4
    if abs(totang-2*np.pi)>0.3:
        return 0
    else:
        return 1


def test_0318_t1():
    filename="/time_RGB/0318/t1/1020-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t1/1507-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t1/1993-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)
    
    filename="/time_RGB/0318/t1/2502-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t1/2996-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t1/3505-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    # ###################################################
    filename="/time_RGB/0318/t2/1006-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t2/1503-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t2/2002-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)
    
    filename="/time_RGB/0318/t2/2543-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t2/2999-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t2/3508-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    ###################################################
    filename="/time_RGB/0318/t3/0999-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t3/1509-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t3/2016-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)
    
    filename="/time_RGB/0318/t3/2510-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t3/3058-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t3/3503-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)


def test_0318_t4():
    filename="/time_RGB/0318/t4/1364-0318-1.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/1364-0318-2.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/1364-0318-3.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/2208-0318-1.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/2208-0318-2.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/2208-0318-3.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/2753-0318-1.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/2753-0318-2.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/2753-0318-3.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/3177-0318-1.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/3177-0318-2.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t4/3177-0318-3.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

def test_0318_t5():
    filename="/time_RGB/0318/t5/1807-17079-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/1807-17692-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/1807-18348-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/2648-16701-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/2648-17346-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/2648-18151-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/2648-18770-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/3440-16445-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/3440-17029-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/3440-17587-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/3440-18090-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/3440-18548-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t5/3440-19026-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)


def test_0318_t6():
    filename="/time_RGB/0318/t6/1778-1464-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t6/1981-1827-0318.png"   
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t6/2115-2282-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)
 
    filename="/time_RGB/0318/t6/2662-2842-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0318/t6/3003-2784-0318.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

def test_0326_t1():
    filename="/time_RGB/0326/1.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0326/2.png"   
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

    filename="/time_RGB/0326/3.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)
    cv2.waitKey(0)

if __name__ == '__main__':
    # filename="/time_RGB/15m0312.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findCanny(img,"r")
    # findCanny(img,"g")
    # findCanny(img,"b")
    # cv2.polylines(img, [r,g,b], True, (255, 0, 0), 1)
    # cv2.imshow("a",img)
    # cv2.waitKey(0) 
    # print(HSVrang)


    # filename="/time_RGB/1m0315.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/15m0315.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/2m0315.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/25m0315.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/3m0315.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/35m0315.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)


    # filename="/time_RGB/4m0315.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/1m0312-12.png" 
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)


    # filename="/time_RGB/2m0312-12.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    filename="/time_RGB/2m0312-12-45.png"
    img = cv2.imread(os.getcwd()+filename)
    img = cv2.imread("/home/yuqiang/Pictures/Screenshot from 2021-08-30 17-47-28.png") 
    findRGB(img)
    cv2.imshow("a",img)
    cv2.waitKey(0)    
 
    # filename="/time_RGB/2m0312-12+45.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/0318/t6/1778-1464-0318.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)
    

    # filename="/time_RGB/0318/t6/1981-1827-0318.png"   
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/0318/t6/2115-2282-0318.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)
 
    # filename="/time_RGB/0318/t6/2662-2842-0318.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)

    # filename="/time_RGB/0318/t6/3003-2784-0318.png"
    # img = cv2.imread(os.getcwd()+filename)
    # findRGB(img)




    # filename="/time_RGB/0326/1.png"
    # img1 = cv2.imread(os.getcwd()+filename)
    # # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) 


    # filename="/time_RGB/0326/2.png"   
    # img2 = cv2.imread(os.getcwd()+filename)
    # # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV) 

        

    # filename="/time_RGB/0326/3.png"
    # img3 = cv2.imread(os.getcwd()+filename)
    # # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV) 
    # t=time.time()
    # import csv
    # t_list=[0 for i in range(1000)]
    # for i in range(1000):
    #     findRGB(img1)

    #     findRGB(img2)

    #     findRGB(img3)
    #     t_list[i]=time.time()-t
    #     t=time.time()
    # print("========================")
    # with open('output2.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for i in range(1000):
    #         writer.writerow([t_list[i]])

    #   test mulit target
    # import mulitTarget

    # filename="/home/yuqiang/catkin_ws4/src/recent_test_case/4target.png"
    # img = cv2.imread(filename)
    # xyid,ip = mulitTarget.find_aruco_mean(img)
    # img_set = mulitTarget.divImg(ip,img)
    # for i in img_set:
    #     cv2.imshow("q2",i)
    #     cv2.waitKey(0)
    # pool=ThreadPool()
    # t=time.time()
    # import csv
    
    # t_list=[0 for i in range(1000)]
    # for i in range(1000):
    #     # res = pool.map(findRGB, img_set)
    #     findRGB(img_set[0])
    #     findRGB(img_set[1])
    #     findRGB(img_set[2])
    #     findRGB(img_set[3])
    #     t_list[i]=time.time()-t
    #     t=time.time()
    # print("========================")
    # with open('output4.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for i in range(1000):
    #         writer.writerow([t_list[i]])