#!/usr/bin/env python
import os
import rospy
from std_msgs.msg import Float32
import time


def cb_fun(data):
    print(data.data,"1")
    time.sleep(10)
    print(data.data,"2")
rospy.init_node('sdsds', anonymous=True)
sssub = rospy.Subscriber('rrrrr', Float32, cb_fun)


rospy.spin()