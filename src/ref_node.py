#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
import time

def cb_takeoff(data):
    global is_takeoff
    is_takeoff=1

def cb_land(data):
    global is_takeoff
    is_takeoff=0   




rospy.init_node('ref_node', anonymous=True)
takeoff_sub = rospy.Subscriber('tello/takeoff', Empty, cb_takeoff)
ref_pub = rospy.Publisher('ref', Twist, queue_size=1)
land_sub = rospy.Subscriber('tello/land', Empty, cb_land)
rate = rospy.Rate(20)


is_takeoff=0
while  not rospy.is_shutdown():
    if is_takeoff:
        ref_pub_msg=Twist()
        ref_pub_msg.linear.x = 2
        ref_pub_msg.linear.y = 0
        ref_pub_msg.linear.z = 0
        box_pub.publish(ref_pub_msg)


    rate.sleep()