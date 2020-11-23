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




rospy.init_node('ref_node', anonymous=True)
takeoff_sub = rospy.Subscriber('tello/takeoff', Empty, cb_takeoff)
ref_pub = rospy.Publisher('ref', Twist, queue_size=1)
land_sub = rospy.Subscriber('tello/land', Empty, cb_land)
rate = rospy.Rate(20)



m=0

while  not rospy.is_shutdown():
    if is_takeoff:
        print(t)
        if m==0:
            if t>=25:
                m=1
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 3
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0
                ref_pub.publish(ref_pub_msg)
        if m==1:
            if t>=10:
                m=2
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 4
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0
                ref_pub.publish(ref_pub_msg)
        if m==2:
            if t>=10:
                m=3
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 5
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0
                ref_pub.publish(ref_pub_msg)
        if m==3:
            if t>=20:
                m=4
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 6
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0
                ref_pub.publish(ref_pub_msg)
        if m==4:
            if t>=20:
                m=5
                t=float(0)
            else:
                t=t+0.05
                ref_pub_msg=Twist()
                ref_pub_msg.linear.x = 7
                ref_pub_msg.linear.y = 0
                ref_pub_msg.linear.z = 0
                ref_pub.publish(ref_pub_msg)
    rate.sleep()