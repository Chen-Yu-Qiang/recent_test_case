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

while 1:
    print("t=takeoff\nl=land\n>>>")
    op=raw_input()
    if op == "t":
        rospy.loginfo("takeoff!")
        #takeoff_pub.publish(Empty())
    elif op=="l":
        rospy.loginfo("land!!!")
        #land_pub.publish(Empty())