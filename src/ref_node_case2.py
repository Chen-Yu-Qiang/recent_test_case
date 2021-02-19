#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import time
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
                ref_pub_msg.linear.x = 1
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0.9
                ref_pub.publish(ref_pub_msg)
        if m==1:
            if t>=30:
                m=2
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0.9
                ref_pub.publish(ref_pub_msg)
        if m==2:
            if t>=30:
                m=2.5
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0.9
                ref_pub.publish(ref_pub_msg)
        if m==2.5:
            if t>=30:
                m=3
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0.9
                ref_pub.publish(ref_pub_msg)
        if m==3:
            if t>=30:
                m=4
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 1
                ref_pub_msg.linear.y = 0+r2g_msg.linear.y
                ref_pub_msg.linear.z = 0.9+r2g_msg.linear.z
                ref_pub.publish(ref_pub_msg)
        if m==4:
            if t>=30:
                m=5
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2
                ref_pub_msg.linear.y = 0+r2g_msg.linear.y
                ref_pub_msg.linear.z = 0.9+r2g_msg.linear.z
                ref_pub.publish(ref_pub_msg)
        if m==5:
            if t>=30:
                m=5.5
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.y = 0+r2g_msg.linear.y
                ref_pub_msg.linear.z = 0.9+r2g_msg.linear.z
                ref_pub.publish(ref_pub_msg)
        if m==5.5:
            if t>=30:
                m=6
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2
                ref_pub_msg.linear.y = 0+r2g_msg.linear.y
                ref_pub_msg.linear.z = 0.9+r2g_msg.linear.z
                ref_pub.publish(ref_pub_msg)
        if m==6:
            if t>=30:
                m=7
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 1
                ref_pub_msg.linear.y = 0+r2b_msg.linear.y
                ref_pub_msg.linear.z = 0.9+r2b_msg.linear.z
                ref_pub.publish(ref_pub_msg)
        if m==7:
            if t>=30:
                m=8
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 2
                ref_pub_msg.linear.y = 0+r2b_msg.linear.y
                ref_pub_msg.linear.z = 0.9+r2b_msg.linear.z
                ref_pub.publish(ref_pub_msg)
        if m==8:
            if t>=30:
                m=9
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.y = 0+r2b_msg.linear.y
                ref_pub_msg.linear.z = 0.9+r2b_msg.linear.z
                ref_pub.publish(ref_pub_msg)
    rate.sleep()