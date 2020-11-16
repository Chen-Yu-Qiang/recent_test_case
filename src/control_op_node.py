#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import threading
import time


rospy.init_node('control_op_node', anonymous=True)
takeoff_pub = rospy.Publisher('tello/takeoff', Empty, queue_size=1)
land_pub = rospy.Publisher('tello/land', Empty, queue_size=1)
isSIM=rospy.get_param('isSIM')
while 1:
    print("t=takeoff\nl=land\n>>>")
    op=raw_input()
    if op == "t":
        rospy.loginfo("takeoff!")
        if isSIM==0:
            takeoff_pub.publish(Empty())
            rospy.loginfo("real takeoff!")
    else:
        rospy.loginfo("land!!!")
        if isSIM==0:
            land_pub.publish(Empty())
            rospy.loginfo("real land!!!")