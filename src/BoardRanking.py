#!/usr/bin/env python
import numpy as np
from geometry_msgs.msg import Twist



marker_set=[[0,0,1.5*np.pi]]
def P(Drone,Marker):
    # Drone=[x,y,theta]
    # Marker=[x,y,theta]
    #Drone=[Drone_msg.linear.x,Drone_msg.linear.y,Drone_msg.angular.z]
    theta1=abs(np.arctan2((Drone[1]-Marker[1]),(Drone[0]-Marker[0]))+np.pi/2-Drone[2])

    theta2=abs(Marker[2]-Drone[2]-np.pi)

    return theta1,theta2

def ranking(Drone_msg):
    global marker_set
    Drone=[Drone_msg.linear.x,Drone_msg.linear.y,Drone_msg.angular.z]
    m=0
    ms,_=P(Drone,marker_set[0])
    for i in range(1,len(marker_set)):
        s1,_=P(Drone,marker_set[i])
        if s1<ms:
            m=i
            ms=s1
    return m


def gotoOrg(i,data):
    global marker_set
    box_org_pub_msg = Twist()
    box_org_pub_msg.linear.x = data.linear.x+marker_set[i][0]
    box_org_pub_msg.linear.y = data.linear.y+marker_set[i][1]
    box_org_pub_msg.linear.z = data.linear.z
    return box_org_pub_msg