#!/usr/bin/env python
import cv2
import os
import numpy as np
import time
from cv2 import aruco
from geometry_msgs.msg import Twist


def find_aruco_mean(img):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict)
    
    # print(corners, ids)
    if ids is None:
        return -1,-1
    # else:
        # print(ids)
    x=[0 for i in range(len(corners))]
    y=[0 for i in range(len(corners))]
    idss=[0 for i in range(len(corners))]
    xyid=[0 for i in range(len(corners))]
    max_x=[0 for i in range(len(corners))]
    min_x=[960 for i in range(len(corners))]
    for i in range(len(corners)):
        for j in range(4):
            x[i]=x[i]+corners[i][0][j][0]/4
            y[i]=y[i]+corners[i][0][j][1]/4
            max_x[i]=max(max_x[i],corners[i][0][j][0])
            min_x[i]=min(min_x[i],corners[i][0][j][0])
        idss[i]=ids[i][0]

        # print(xyid)
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
            ip[i]=min(ip[i],960)
        s_ip=max(0,xyid[0][0]-(xyid[0][0]-min(min_x))*6)
        xyid=xyid[0:-1]
        ip=[s_ip]+ip
        # print(ip)
    return xyid,ip

def divImg(ip,img):
    img_set=[img for i in range(len(ip)-1)]
    if len(ip)>=1:
        for i in range(len(ip)-1):
            sss=np.zeros([720,960],dtype=np.uint8)
            sss[0:720,int(ip[i]):int(ip[i+1])]=255
            img_set[i]=cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=sss)

    return img_set


def TwoTargetPos(tar1,tar2,rel=None):
    """
      (board)      (UAV)      ^ y-axis
        |          o   o      |
    <---|         <  X        |---->  x-axis     (Definition relative to the target)
        |          o   o

    """
    if rel is None:
        rel=Twist()
        rel.linear.x=3
        rel.linear.y=0
        rel.linear.z=0
        rel.angular.z=0
    ang=(tar1.angular.z+tar2.angular.z)/2.0
    cen_x=(tar1.linear.x+tar2.linear.x)/2.0
    cen_y=(tar1.linear.y+tar2.linear.y)/2.0
    cen_z=(tar1.linear.z+tar2.linear.z)/2.0
    out_msg=Twist()
    out_msg.linear.x=cen_x-np.sin(ang-np.pi)*rel.linear.x-np.sin(ang-0.5*np.pi)*rel.linear.y
    out_msg.linear.y=cen_y+np.cos(ang-np.pi)*rel.linear.x+np.cos(ang-0.5*np.pi)*rel.linear.y
    out_msg.linear.z=cen_z+rel.linear.z
    out_msg.angular.z=ang+rel.angular.z

    return out_msg

def v_y(x,y,th,l=-0.1):
    xx=x+np.cos(th+np.pi*0.5)*l
    yy=y+np.sin(th+np.pi*0.5)*l
    return [x,xx],[y,yy]


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
    filename="/home/yuqiang/catkin_ws4/src/recent_test_case/3target.png"
    img = cv2.imread(filename)
    cv2.imshow("q",img)
    xyid,ip = find_aruco_mean(img)
    print(xyid,ip)
    img_set=divImg(ip,img)
    for i in img_set:
        cv2.imshow("q2",i)
        cv2.waitKey(0)
    filename="/home/yuqiang/catkin_ws4/src/recent_test_case/4target.png"
    img = cv2.imread(filename)
    cv2.imshow("q",img)
    xyid,ip = find_aruco_mean(img)
    print(xyid,ip)
    img_set=divImg(ip,img)
    for i in img_set:
        cv2.imshow("q2",i)
        cv2.waitKey(0)



    # import random
    # target1=Twist()
    # target1.linear.x=(random.random()-0.5)*3
    # target1.linear.y=(random.random()-0.5)*3
    # target1.linear.z=0.4
    # target1.angular.z=np.pi*2*random.random()
    # target2=Twist()
    # target2.linear.x=(random.random()-0.5)*3
    # target2.linear.y=(random.random()-0.5)*3
    # target2.linear.z=0.4
    # target2.angular.z=np.pi*2*random.random()




    # target1=Twist()
    # target1.linear.x=0
    # target1.linear.y=0
    # target1.linear.z=0
    # target1.angular.z=np.pi*0.5
    # target2=Twist()
    # target2.linear.x=-1
    # target2.linear.y=1
    # target2.linear.z=0
    # target2.angular.z=np.pi*0.5
    
    
    # rel=Twist()
    # rel.linear.x=2
    # rel.linear.y=0
    # rel.linear.z=0
    # rel.angular.z=0
    # res=TwoTargetPos(target1,target2,rel)
    # print(res)
    # import matplotlib.pyplot as plt
    # plt.scatter(target1.linear.x,target1.linear.y)
    # [x,xx],[y,yy]=v_y(target1.linear.x,target1.linear.y,target1.angular.z)
    # plt.plot([x,xx],[y,yy])
    # plt.scatter(target2.linear.x,target2.linear.y)
    # [x,xx],[y,yy]=v_y(target2.linear.x,target2.linear.y,target2.angular.z)
    # plt.plot([x,xx],[y,yy])
    # # plt.scatter((target2.linear.x+target1.linear.x)*0.5,(target2.linear.y+target1.linear.y)*0.5)
    # # plt.plot([target1.linear.x,target2.linear.x],[target1.linear.y,target2.linear.y])
    # plt.scatter(res.linear.x,res.linear.y)
    # [x,xx],[y,yy]=v_y(res.linear.x,res.linear.y,res.angular.z,0.3)
    # plt.plot([x,xx],[y,yy])    
    # # plt.axis([-3,3,-3,3])
    # plt.grid(True)

    # plt.show()


 