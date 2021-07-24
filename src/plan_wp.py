#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
import multiTarget
class target:
    def __init__(self,_i):
        self.sub = rospy.Subscriber('target'+str(_i), Twist, self.cb_fun)
        self.i=_i
        self.data=None
    def cb_fun(self,_data):
        self.data=_data


rospy.init_node('plan_node', anonymous=True)
plan_wp_pub = rospy.Publisher('plan_wp', Twist, queue_size=1)
target_set=[target(i) for i in range(51,60)]
rate = rospy.Rate(30)
while  not rospy.is_shutdown():

    res=multiTarget.TwoTargetPos(target_set[0].data,target_set[1].data)
    plan_wp_pub.publish(plan_wp_pub)
    rate.sleep()