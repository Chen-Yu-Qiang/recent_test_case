#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
import mulitTarget
import time
import filter_lib
class target:
    def __init__(self,_i):
        self.sub = rospy.Subscriber('target'+str(_i), Twist, self.cb_fun)
        self.i=_i
        self.data=None
        self.mfilter=filter_lib.meanFilter(11)
        self.after_filter_pub=rospy.Publisher('target'+str(_i)+"_f", Twist, queue_size=1)
        self.t=time.time()
    def cb_fun(self,_data):
        self.data=self.mfilter.update(_data)
        self.after_filter_pub.publish(self.data)
        self.t=time.time()
    def isTimeout(self):
        if (time.time()-self.t<5) and (not (self.data is None)):
            return 0
        else:
            return 1

rospy.init_node('plan_node', anonymous=True)
plan_wp_pub = rospy.Publisher('plan_wp', Twist, queue_size=1)
target_set=[target(i) for i in range(51,60)]
rate = rospy.Rate(30)
while  not rospy.is_shutdown():
    if target_set[0].isTimeout() and target_set[1].isTimeout() :
        pass
    elif target_set[0].isTimeout() and (not target_set[1].isTimeout()):
        res=target_set[1].data
        plan_wp_pub.publish(res)
    elif (not target_set[0].isTimeout()) and (target_set[1].isTimeout()):
        res=target_set[0].data
        plan_wp_pub.publish(res)
    elif (not target_set[0].isTimeout()) and (not target_set[1].isTimeout()):
        res=mulitTarget.TwoTargetPos(target_set[0].data,target_set[1].data)
        plan_wp_pub.publish(res)
    rate.sleep()