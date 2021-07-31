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

Target51_position=Twist()
def cb_target51(data):
    global Target51_position
    Target51_position=data

Target52_position=Twist()
def cb_target52(data):
    global Target52_position
    Target52_position=data

Target_all_position=Twist()
def cb_target_all(data):
    global Target_all_position
    Target_all_position=data

def cheak_ang_range(i):
    if i>np.pi/2*3:
        i=i-2*np.pi
        return cheak_ang_range(i)
    elif i<np.pi*(-0.5):
        i=i+np.pi*2
        return cheak_ang_range(i)
    else:
        return i


def wantPos(Target_position,dis):
    want_position=Twist()
    want_position.linear.x = Target_position.linear.x+np.sin(Target_position.angular.z)*dis
    want_position.linear.y = Target_position.linear.y-np.cos(Target_position.angular.z)*dis
    want_position.linear.z = Target_position.linear.z+0.9
    want_position.angular.z = Target_position.angular.z
    return want_position

def p2p_mean(p1,p2,t2,t):
    '''
    From one point to another, go smoothly

    Args:
        p1 (Twist): from point
        p2 (Twist): to point
        t2 (floot): total time
        t (float): now time
    Retunes:
        (Twist): at the time 
    '''
    if t>=t2:
        return p2
    if t<0:
        return p1
    out_msg=Twist()
    out_msg.linear.x=1.0*(p2.linear.x-p1.linear.x)/t2*t+p1.linear.x
    out_msg.linear.y=1.0*(p2.linear.y-p1.linear.y)/t2*t+p1.linear.y
    out_msg.linear.z=1.0*(p2.linear.z-p1.linear.z)/t2*t+p1.linear.z
    out_msg.angular.z=1.0*(p2.angular.z-p1.angular.z)/t2*t+p1.angular.z
    return out_msg

rospy.init_node('ref_node_case2', anonymous=True)
takeoff_sub = rospy.Subscriber('tello/takeoff', Empty, cb_takeoff)
ref_pub = rospy.Publisher('ref', Twist, queue_size=1)
land_pub = rospy.Publisher('tello/land', Empty, queue_size=1)
land_sub = rospy.Subscriber('tello/land', Empty, cb_land)
target51_sub = rospy.Subscriber('target51', Twist, cb_target51)
target52_sub = rospy.Subscriber('target52', Twist, cb_target52)
target_all_sub = rospy.Subscriber('plan_wp', Twist, cb_target_all)
Ts=0.01
rate = rospy.Rate(1/Ts)


m=0
times=0
while  not rospy.is_shutdown():
    if is_takeoff:
        # print(t)
        if m==0:
            if t>=20:
                m=1
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 1.5
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
                ref_pub_msg.linear.x = 2
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 1.5
                ref_pub_msg.angular.z = 90.0/57.296
                ref_pub.publish(ref_pub_msg)
        if m==2:
            if t>=20:
                m=3
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2.5
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 1.5
                ref_pub_msg.angular.z = 90.0/57.296
                ref_pub.publish(ref_pub_msg)
        if m==3:
            if t>=20:
                m=4
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 1.5
                ref_pub_msg.angular.z = 90.0/57.296
                ref_pub.publish(ref_pub_msg)
        if m==4:
            if t>=20:
                m=100
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 3.5
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 1.5
                ref_pub_msg.angular.z = 90.0/57.296
                ref_pub.publish(ref_pub_msg)
        if m==5:
            if t>=5:
                m=6
                t=float(0)
            else:
                t=t+Ts
                p1=Twist()
                p1.linear.x = 3.5
                p1.linear.y = 0
                p1.linear.z = 1.5
                p1.angular.z = 90.0/57.296
                ref_pub_msg=Twist()
                ref_pub_msg=p2p_mean(p1,wantPos(Target51_position,2),5,t)
                ref_pub.publish(ref_pub_msg)
        if m==6:
            if t>=20:
                m=7
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=wantPos(Target51_position,2)
                ref_pub.publish(ref_pub_msg)
        if m==7:
            if t>=5:
                m=8
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Twist()
                ref_pub_msg=p2p_mean(wantPos(Target51_position,2),wantPos(Target52_position,2),5,t)
                ref_pub.publish(ref_pub_msg)
        if m==8:
            if t>=20:
                m=9
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=wantPos(Target52_position,2)
                ref_pub.publish(ref_pub_msg)
        if m==9:
            if t>=20:
                m=5
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=p2p_mean(wantPos(Target52_position,2),Target_all_position,5,t)
                ref_pub.publish(ref_pub_msg)
        if m==10:
            if t>=20:
                m=100
                t=float(0)
            else:
                t=t+Ts
                ref_pub_msg=Target_all_position
                ref_pub.publish(ref_pub_msg)
        if m==100:
            land_pub.publish(Empty())
            m=101
    rate.sleep()