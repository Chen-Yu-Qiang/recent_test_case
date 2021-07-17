#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import time
import numpy as np
t=float(0)
def cb_takeoff(data):
    global is_takeoff,t
    is_takeoff=1
    t=float(0)

def cb_land(data):
    global is_takeoff
    is_takeoff=0   
isSIM=rospy.get_param('isSIM')
is_takeoff=0
if isSIM==1:
    is_takeoff=1
else:
    is_takeoff=0

Target_position=Twist()
def cb_target(data):
    global Target_position
    Target_position=data
    Target_position.angular.z=np.pi

def cheak_ang_range(i):
    if i>np.pi/2*3:
        i=i-2*np.pi
        return cheak_ang_range(i)
    elif i<np.pi*(-0.5):
        i=i+np.pi*2
        return cheak_ang_range(i)
    else:
        return i


def wantPos(Target_position):
    want_position=Twist()
    want_position.linear.x = Target_position.linear.x+np.sin(Target_position.angular.z)*1.5
    want_position.linear.y = Target_position.linear.y-np.cos(Target_position.angular.z)*1.5
    want_position.linear.z = Target_position.linear.z+0.9
    want_position.angular.z = Target_position.angular.z
    return want_position

rospy.init_node('ref_node_case2', anonymous=True)
takeoff_sub = rospy.Subscriber('tello/takeoff', Empty, cb_takeoff)
ref_pub = rospy.Publisher('ref', Twist, queue_size=1)
land_pub = rospy.Publisher('tello/land', Empty, queue_size=1)
land_sub = rospy.Subscriber('tello/land', Empty, cb_land)
target_sub = rospy.Subscriber('target', Twist, cb_target)
Ts=0.01
rate = rospy.Rate(1/Ts)


m=0
times=0
while  not rospy.is_shutdown():
    if is_takeoff:
        print(t)
        if m==0:
            if t>=30:
                m=1
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2.5
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 1.5
                ref_pub_msg.angular.z = 90.0/57.296
                ref_pub.publish(ref_pub_msg)
        if m==1:
            if t>=20:
                m=2
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2.5
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 1.5
                ref_pub_msg.angular.z = (90.0+(90.0/20.0*t))/57.296
                ref_pub.publish(ref_pub_msg)
        if m==2:
            if t>=5:
                m=3
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                want_position=wantPos(Target_position)
                ref_pub_msg.linear.x = (want_position.linear.x-2.5)/5*t+2.5
                ref_pub_msg.linear.y = (want_position.linear.y-0)/5*t+0.0
                ref_pub_msg.linear.z = (want_position.linear.z-1.5)/5*t+1.5
                ref_pub_msg.angular.z = 180.0/57.296
                ref_pub.publish(ref_pub_msg)
        if m==3:
            if t>=60:
                m=100
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                want_position=wantPos(Target_position)
                ref_pub_msg.linear.x = want_position.linear.x
                ref_pub_msg.linear.y = want_position.linear.y
                ref_pub_msg.linear.z = want_position.linear.z
                ref_pub_msg.angular.z = want_position.angular.z
                ref_pub.publish(ref_pub_msg)
        if m==100:
            land_pub.publish(Empty())
            m=101
    rate.sleep()