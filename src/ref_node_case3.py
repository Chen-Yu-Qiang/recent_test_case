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
r2g_msg=Twist()
r2b_msg=Twist()
def cb_from_box_r2g(data):
    global r2g_msg
    r2g_msg=data

def cb_from_box_r2b(data):
    global r2b_msg
    r2b_msg=data


def cheak_ang_range(i):
    if i>np.pi/2*3:
        i=i-2*np.pi
        return cheak_ang_range(i)
    elif i<np.pi*(-0.5):
        i=i+np.pi*2
        return cheak_ang_range(i)
    else:
        return i


rospy.init_node('ref_node_case2', anonymous=True)
takeoff_sub = rospy.Subscriber('tello/takeoff', Empty, cb_takeoff)
from_box_r2g_sub = rospy.Subscriber('from_box_r2g', Twist, cb_from_box_r2g)
from_box_r2b_sub = rospy.Subscriber('from_box_r2b', Twist, cb_from_box_r2b)
ref_pub = rospy.Publisher('ref', Twist, queue_size=1)
land_sub = rospy.Subscriber('tello/land', Empty, cb_land)
rate = rospy.Rate(30)



m=0

while  not rospy.is_shutdown():
    if is_takeoff:
        print(t)
        if m==0:
            if t>=30:
                m=1
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0.9
                ref_pub_msg.angular.z = cheak_ang_range(90.0/57.296)
                ref_pub.publish(ref_pub_msg)
        if m==1:
            if t>=30:
                m=2
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                # ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0.9
                ref_pub_msg.angular.z = cheak_ang_range(90.0/57.296)
                ref_pub.publish(ref_pub_msg)
        if m==2:
            if t>=25:
                m=3
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                # ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.x = 3*np.cos(t*2/57.296)
                ref_pub_msg.linear.y = 3*np.sin(t*2/57.296)
                ref_pub_msg.linear.z = 0.9
                # ref_pub_msg.angular.z = 70.0/57.296
                ref_pub_msg.angular.z = cheak_ang_range((90.0+t*2)/57.296)
                ref_pub.publish(ref_pub_msg)
        if m==3:
            if t>=30:
                m=4
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 1.94
                ref_pub_msg.linear.y = 2.28
                ref_pub_msg.linear.z = 0.9
                # ref_pub_msg.angular.z = 90.0/57.296
                ref_pub_msg.angular.z = cheak_ang_range(140/57.296)
                ref_pub.publish(ref_pub_msg)
        if m==4:
            if t>=25:
                m=5
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 3*np.cos((25-t)*2/57.296)
                ref_pub_msg.linear.y = 3*np.sin((25-t)*2/57.296)
                ref_pub_msg.linear.z = 0.9
                # ref_pub_msg.angular.z = 90.0/57.296
                ref_pub_msg.angular.z = cheak_ang_range((90.0+(25-t)*2)/57.296)
                ref_pub.publish(ref_pub_msg)
        if m==5:
            if t>=30:
                m=6
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 1.94
                ref_pub_msg.linear.y = -2.28
                ref_pub_msg.linear.z = 0.9
                ref_pub_msg.angular.z = cheak_ang_range(40/57.296)
                ref_pub.publish(ref_pub_msg)
        if m==6:
            if t>=25:
                m=7
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                # ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.x = 3*np.cos((t-25)*2/57.296)
                ref_pub_msg.linear.y = 3*np.sin((t-25)*2/57.296)
                ref_pub_msg.linear.z = 0.9
                # ref_pub_msg.angular.z = 70.0/57.296
                ref_pub_msg.angular.z = cheak_ang_range((90.0+(t-25)*2)/57.296)
                ref_pub.publish(ref_pub_msg)
       
    rate.sleep()