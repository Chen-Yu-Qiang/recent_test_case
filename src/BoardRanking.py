#!/usr/bin/env python
import numpy as np
from geometry_msgs.msg import Twist
from cv2 import aruco


marker_set=[[0,0,1.5*np.pi],[0,-9,1.5*np.pi]]
def P(Drone,Marker):
    # Drone=[x,y,theta]
    # Marker=[x,y,theta]
    #Drone=[Drone_msg.linear.x,Drone_msg.linear.y,Drone_msg.angular.z]
    Marker[2]=Marker[2]%(2*np.pi)
    theta1=abs(np.arctan2((Drone[1]-Marker[1]),(Drone[0]-Marker[0]))+np.pi/2-Drone[2])
    if theta1>np.pi:
        theta1=2*np.pi-theta1
    theta2=abs(Marker[2]-Drone[2]-np.pi)
    d=np.sqrt((Drone[0]-Marker[0])*(Drone[0]-Marker[0])+(Drone[1]-Marker[1])*(Drone[1]-Marker[1]))
    return [theta1,theta2,d]

def ranking(Drone_msg):
    global marker_set
    Drone=[Drone_msg.linear.x,Drone_msg.linear.y,Drone_msg.angular.z]
    m=-1
    s_all=[0 for i in range(len(marker_set))]
    # ms=41.0/180*np.pi

    ms=1000
    for i in range(0,len(marker_set)):
        s_all[i]=P(Drone,marker_set[i])
        # if s_all[i][0]<ms and (s_all[i][1]<np.pi/2 or s_all[i][1]>np.pi*3/2) and s_all[i][2]<5:
        if s_all[i][0]<ms:
            m=i
            ms=s_all[i][0]
    return m

def checkAruco(img):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict)
    if len(ids)==0:
        return -1
    else:
        return ids[0]

def gotoOrg(i,data):
    global marker_set
    box_org_pub_msg = data
    box_org_pub_msg.linear.x = data.linear.x+marker_set[i][0]
    box_org_pub_msg.linear.y = data.linear.y+marker_set[i][1]
    print(marker_set[i][0],marker_set[i][1])
    return box_org_pub_msg


if __name__ == '__main__':
    pass
    # for ii in range(0,360):
    #     i=np.pi/180*ii
    #     Drone=Twist()
    #     Drone.linear.x=np.cos(i)*2
    #     Drone.linear.y=np.sin(i)*2
    #     Drone.angular.z=np.pi/2+i
    #     print(i*180/np.pi,ranking(Drone))

    # for ii in range(0,360):
    #     i=np.pi/180*ii
    #     Drone=Twist()
    #     Drone.linear.x=np.cos(i)*2
    #     Drone.linear.y=np.sin(i)*2
    #     Drone.angular.z=np.pi/2
    #     print(i*180/np.pi,ranking(Drone))

    # for yy in range(-200,100,1):
    #     y=yy/10.0
    #     Drone=Twist()
    #     Drone.linear.x=2
    #     Drone.linear.y=y
    #     Drone.angular.z=np.pi/2
    #     print(y,ranking(Drone))

    # for xx in range(-200,100,1):
    #     x=xx/10.0
    #     Drone=Twist()
    #     Drone.linear.x=x
    #     Drone.linear.y=0
    #     Drone.angular.z=np.pi/2
    #     print(x,ranking(Drone))