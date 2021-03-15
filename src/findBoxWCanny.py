#!/usr/bin/env python
import cv2
import os
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
r=None
g=None
b=None
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

def findRect(img,color):
    global HSVrang
    tm_hour=10

    # convert to HSV

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    # print(hsv[360][480])
    if color=="g":
        if tm_hour==25:
            lower_g = np.array(HSVrang["gL"])
            upper_g = np.array(HSVrang["gH"])

        elif tm_hour<=18 and tm_hour>=6 :
            lower_g = np.array([67, 180, 30])
            upper_g = np.array([78, 253, 118])
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
            lower_red = np.array([174, 165, 35])
            upper_red = np.array([179, 255, 164])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 165, 35])
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
        cv2.imshow('mask'+str(color), mask)
        cv2.waitKey(1)

    if color=="b":

        if tm_hour==25:
            lower_b = np.array(HSVrang["bL"])
            upper_b = np.array(HSVrang["bH"])

        elif tm_hour<=18 and tm_hour>=6 :
            lower_b = np.array([80, 165, 10])
            upper_b = np.array([117, 250, 120])
        else:
            lower_b = np.array([108, 126, 124])
            upper_b = np.array([116, 175, 161])
            lower_b = np.array([103, 126, 124])
            upper_b = np.array([116, 210, 180])
        mask=cv2.inRange(hsv, lower_b, upper_b)
    
    # cv2.imshow('mask'+str(color), mask)
    # cv2.waitKey(0)
    mask = cv2.dilate(mask, np.ones((17,17), np.uint8), iterations = 1)
    result = cv2.bitwise_and(img, img, mask=mask)

    kernel = np.ones((17,17), np.uint8)
    erosion = cv2.dilate(result, kernel, iterations = 1)
    gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow('erosion'+str(color), erosion)
    cv2.waitKey(1)

    _,contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    A_max=0
    c_max=None
    []
    for c in contours:
        __, _, w1, h1 = cv2.boundingRect(c)
        #print(cv2.boundingRect(c))
        if cv2.contourArea(c)>A_max and w1>50 and h1>70 :
            A_max=cv2.contourArea(c)
            c_max=c

    x, y, w, h = cv2.boundingRect(c_max)
    # print(x,y,w,h)
   
    return x,y,w,h

def findCanny(img,color):
    global r,g,b
    bigger=0
    # cv2.imshow("org"+str(color),cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # cv2.waitKey(0)
    x,y,w,h = findRect(img,color)
    
    if x*y*w*h==0:
        # print("findRect=0",color)
        return
    mask = np.zeros((720,960,1), np.uint8) 
    mask.fill(0)
    cv2.rectangle(mask, (int(x-w*bigger), int(y-h*bigger)), (int(x+w*(1+2*bigger)), int(y+h*(1+2*bigger))), 255, -1)


    image = cv2.bitwise_or(img, img, mask=mask)

    #image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    # print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('blurred', blurred)
    y_min=max(0,int(y-h*bigger+1))
    y_max=min(719,int(y+h*(1+2*bigger)-1))
    x_min=max(0,int(x-w*bigger+1))
    x_max=min(959,int(x+w*(1+2*bigger)-1))
    canny=blurred[y_min:y_max,x_min:x_max]

    # print("G",canny.shape)
    canny = cv2.Canny(canny, 10, 100)

    if canny is None:
        # print("canny is none",color)
        return
    #canny = cv2.GaussianBlur(canny, (5, 5), 0)
    cv2.imshow('canny'+color, canny)
    cv2.waitKey(1)
    # print("F")
    _,contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for ii in range(len(contours)):
        for i in range(len(contours[ii])):
            contours[ii][i]=[[contours[ii][i][0][0]+x_min,contours[ii][i][0][1]+y_min]] 

    cv2.drawContours(image,contours,-1,(0,0,255),-1) 
    cv2.imshow('image'+color, image)
    cv2.waitKey(1)
    A_max=0
    c_max=None
    print(len(contours))
    for c in contours:
        __, _, w1, h1 = cv2.boundingRect(c) 
        #print(cv2.boundingRect(c))
        if cv2.contourArea(c)>A_max and w1>50 and h1>70 :
            A_max=cv2.contourArea(c)
            c_max=c
    if A_max==0:
        # print("no max area",color)
        return
    #print(c_max)
    peri = cv2.arcLength(c_max, True) 
    approx1 = cv2.approxPolyDP(c_max, 0.1*peri, True)
    # print(len(approx1),len(approx1[0]))
    # cv2.polylines(image, [approx1], True, (255, 0, 0), 1)
    # cv2.imshow('imageHSV', image)
    # print(approx1)
    if  not len(approx1)==4:
        print("len(approx1)=",len(approx1),color)
        return
    # print("=== "+str(color)+" good===")
    H_max,S_max,V_max,H_std,S_std,V_std = gethsv(img,approx1)
    # print(color,H_max,S_max,V_max,H_std,S_std,V_std)
    # nowHSV(color,H_max,S_max,V_max,H_std,S_std,V_std)
    # print("")
    if color=="r":
        r=approx1
    elif color=="g":
        g=approx1
    elif color=="b":
        b=approx1
    
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

def xywh(p):
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]=p
    x=(x1*x3*y2 - x2*x3*y1 - x1*x4*y2 + x2*x4*y1 - x1*x3*y4 + x1*x4*y3 + x2*x3*y4 - x2*x4*y3)/(x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)
    y=(x1*y2*y3 - x2*y1*y3 - x1*y2*y4 + x2*y1*y4 - x3*y1*y4 + x4*y1*y3 + x3*y2*y4 - x4*y2*y3)/(x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)
    # (x1,y1)  (x3,y3)
    # (x4,y4)  (x2,y2)
    w=0.5*(x3+x2-x1-x4)
    h=0.5*(y4+y2-y1-y3)
    return int(x),int(y),int(w),int(h)

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
    # plt.plot(img_gray_hist[1:])
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
    manyTime=3
    NowRate=0.01
    if color=="r":
        S_g_low = S_max-manyTime*S_std
        S_g_High = S_max+manyTime*S_std
        V_g_low = V_max-manyTime*V_std
        V_g_High = V_max+manyTime*V_std
        HSVrang["rL1"][1]=int(HSVrang["rL1"][1]*(1-NowRate)+S_g_low*NowRate)
        HSVrang["rL1"][2]=int(HSVrang["rL1"][2]*(1-NowRate)+V_g_low*NowRate)
        HSVrang["rH1"][1]=int(HSVrang["rH1"][1]*(1-NowRate)+S_g_High*NowRate)
        HSVrang["rH1"][2]=int(HSVrang["rH1"][2]*(1-NowRate)+V_g_High*NowRate)
        HSVrang["rL2"][1]=int(HSVrang["rL2"][1]*(1-NowRate)+S_g_low*NowRate)
        HSVrang["rL2"][2]=int(HSVrang["rL2"][2]*(1-NowRate)+V_g_low*NowRate)
        HSVrang["rH2"][1]=int(HSVrang["rH2"][1]*(1-NowRate)+S_g_High*NowRate)
        HSVrang["rH2"][2]=int(HSVrang["rH2"][2]*(1-NowRate)+V_g_High*NowRate)
        if H_max<90:
            H_g_low = H_max-manyTime*H_std/3+180
            H_g_High = H_max+manyTime*H_std/3
        else:
            H_g_low = H_max-manyTime*H_std/3
            H_g_High = H_max+manyTime*H_std/3-180
        HSVrang["rL1"][0]=int(HSVrang["rL1"][0]*(1-NowRate)+H_g_low*NowRate)
        HSVrang["rH2"][0]=int(HSVrang["rH2"][0]*(1-NowRate)+H_g_High*NowRate)
    
    elif color=="g":
        H_g_low = H_max-manyTime*H_std
        H_g_High = H_max+manyTime*H_std
        S_g_low = S_max-manyTime*S_std
        S_g_High = S_max+manyTime*S_std
        V_g_low = V_max-manyTime*V_std
        V_g_High = V_max+manyTime*V_std
        HSVrang["gL"][0]=int(HSVrang["gL"][0]*(1-NowRate)+H_g_low*NowRate)
        HSVrang["gL"][1]=int(HSVrang["gL"][1]*(1-NowRate)+S_g_low*NowRate)
        HSVrang["gL"][2]=int(HSVrang["gL"][2]*(1-NowRate)+V_g_low*NowRate)
        HSVrang["gH"][0]=int(HSVrang["gH"][0]*(1-NowRate)+H_g_High*NowRate)
        HSVrang["gH"][1]=int(HSVrang["gH"][1]*(1-NowRate)+S_g_High*NowRate)
        HSVrang["gH"][2]=int(HSVrang["gH"][2]*(1-NowRate)+V_g_High*NowRate)
    elif color=="b":
        H_g_low = H_max-manyTime*H_std
        H_g_High = H_max+manyTime*H_std
        S_g_low = S_max-manyTime*S_std
        S_g_High = S_max+manyTime*S_std
        V_g_low = V_max-manyTime*V_std
        V_g_High = V_max+manyTime*V_std
        HSVrang["bL"][0]=int(HSVrang["bL"][0]*(1-NowRate)+H_g_low*NowRate)
        HSVrang["bL"][1]=int(HSVrang["bL"][1]*(1-NowRate)+S_g_low*NowRate)
        HSVrang["bL"][2]=int(HSVrang["bL"][2]*(1-NowRate)+V_g_low*NowRate)
        HSVrang["bH"][0]=int(HSVrang["bH"][0]*(1-NowRate)+H_g_High*NowRat)
        HSVrang["bH"][1]=int(HSVrang["bH"][1]*(1-NowRate)+S_g_High*NowRate)
        HSVrang["bH"][2]=int(HSVrang["bH"][2]*(1-NowRate)+V_g_High*NowRate)




def findRGB(img):
    global r,g,b
    r=None
    g=None
    b=None
    
    t=time.time()
    

    # ===========================Multithreading
    # r_jod=threading.Thread(target = findCanny, args = (img,"r"))
    # g_jod=threading.Thread(target = findCanny, args = (img,"g"))
    # b_jod=threading.Thread(target = findCanny, args = (img,"b"))
    # r_jod.start()
    # # print("r start")
    # # g_jod.start()
    # # print("g start")
    # # b_jod.start()
    # # print("b start")
    # r_jod.join()
    # # print("r join")
    # # g_jod.join()
    # # print("g join")
    # # b_jod.join()
    # # print("b join")
    # # print(xywh(div1234(r))) 

    # ==========================Single thread
    findCanny(img,"r")
    findCanny(img,"g")
    findCanny(img,"b")
    if not r is None:
        print("r",xywh(div1234(r)))
    if not g is None:
        print("g",xywh(div1234(g)))
    if not b is None:
        print("b",xywh(div1234(b)))
    cv2.polylines(img, [r,g,b], True, (255, 0, 0), 1)
    # print(time.time()-t)
    cv2.imshow("a",img)
    cv2.waitKey(0)
    # print("r,g,b",r,g,b)
    # print(HSVrang)
    return r,g,b


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


    filename="/time_RGB/1m0312 (copy).png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

    filename="/time_RGB/15m0312.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

    filename="/time_RGB/2m0312.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

    filename="/time_RGB/25m0312.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

    filename="/time_RGB/3m0312.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

    filename="/time_RGB/35m0312.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

    filename="/time_RGB/2m0312-12.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

    filename="/time_RGB/2m0312-12-45.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

    filename="/time_RGB/2m0312-12+45.png"
    img = cv2.imread(os.getcwd()+filename)
    findRGB(img)

