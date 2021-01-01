#!/usr/bin/env python
import os
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry



def cb_box(data):
    global P,X
    ang=12*np.pi/180
    if abs(data.linear.z)>5 or abs(data.linear.y)>5 or abs(data.linear.x)>5 :
        return
    box_x=data.linear.x*np.cos(ang) - data.linear.z*np.sin(ang)
    box_y=data.linear.y
    box_z=data.linear.x*np.sin(ang) + data.linear.z*np.cos(ang)
    
    Z=np.array([[box_x],[box_y],[box_z]])
    Y=Z-np.dot(H1,X)
    S=np.dot(np.dot(H1,P),np.transpose(H1))+R1
    K=np.dot(np.dot(P,np.transpose(H1)),np.linalg.inv(S))
    X=X+np.dot(K,Y)
    P=np.dot(np.eye(6)-np.dot(K,H1),P)

def cb_odom(data):
    global P,X
    ang=12*np.pi/180
    vx = -data.twist.twist.angular.x
    vy = data.twist.twist.angular.y
    vz = -data.twist.twist.angular.z
    
    Z=np.array([[vx],[vy],[vz]])
    Y=Z-np.dot(H2,X)
    S=np.dot(np.dot(H2,P),np.transpose(H2))+R2
    K=np.dot(np.dot(P,np.transpose(H2)),np.linalg.inv(S))
    X=X+np.dot(K,Y)
    P=np.dot(np.eye(6)-np.dot(K,H2),P)


dt=1.0/100
F =np.array(
   [[1,dt,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,dt,0,0],
    [0,0,0,1,0,0],
    [0,0,0,0,1,dt],
    [0,0,0,0,0,1]])

H1=np.array(
   [[1,0,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,0,1,0]])
H2=np.array(
   [[0,1,0,0,0,0],
    [0,0,0,1,0,0],
    [0,0,0,0,0,1]])

Q = np.eye(6)*0.01 
R1 = np.eye(3)*0.001 
R2 = np.eye(3) 
P = np.zeros((6,6))
X=np.zeros((6,1))
X[0]=1.5
rospy.init_node('kf', anonymous=True)
odom_sub = rospy.Subscriber("tello/odom", Odometry, cb_odom)
box_sub = rospy.Subscriber('from_box_merge', Twist, cb_box)
kf_p_pub = rospy.Publisher('from_kf', Twist, queue_size=1)
kf_v_pub = rospy.Publisher('v_kf', Twist, queue_size=1)


rate = rospy.Rate(1.0/dt)
while  not rospy.is_shutdown():


    X = np.dot(F,X)
    P = np.dot(np.dot(F,P),np.transpose(F)) + Q


    kf_p_msg=Twist()
    kf_p_msg.linear.x=X[0]
    kf_p_msg.linear.y=X[2]
    kf_p_msg.linear.z=X[4]
    kf_p_pub.publish(kf_p_msg)
    kf_v_msg=Twist()
    kf_v_msg.linear.x=X[1]
    kf_v_msg.linear.y=X[3]
    kf_v_msg.linear.z=X[5]
    kf_v_pub.publish(kf_v_msg)

    
    rate.sleep()