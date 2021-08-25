#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
import mulitTarget
import time
import filter_lib
import kf_lib
import viewPanning
import numpy as np
dt=1.0/30.0
class target:
    def __init__(self,_i):
        self.sub = rospy.Subscriber('target'+str(_i), Twist, self.cb_fun)
        self.i=_i
        self.data=None
        self.mfilter=filter_lib.meanFilter(11)

        # constantPosition ====================================
        self.kf_x=kf_lib.KalmanFilter(1)
        self.kf_x.constantPosition(dt,0,0.001)
        self.measure_x_p=kf_lib.KF_updater(1,self.kf_x)
        self.measure_x_p.constantPosition_Position(1)

        self.kf_y=kf_lib.KalmanFilter(1)
        self.kf_y.constantPosition(dt,0,0.001)
        self.measure_y_p=kf_lib.KF_updater(1,self.kf_y)
        self.measure_y_p.constantPosition_Position(1)

        self.kf_z=kf_lib.KalmanFilter(1)
        self.kf_z.constantPosition(dt,0,0.001)
        self.measure_z_p=kf_lib.KF_updater(1,self.kf_z)
        self.measure_z_p.constantPosition_Position(1)

        self.kf_th=kf_lib.KalmanFilter(1)
        self.kf_th.constantPosition(dt,1.57,0.005)
        self.measure_th_p=kf_lib.KF_updater(1,self.kf_th)
        self.measure_th_p.constantPosition_Position(1)
        # constantPosition ====================================

        # constantSpeed ====================================
        # self.kf_x=kf_lib.KalmanFilter(2)
        # self.kf_x.constantSpeed(dt,0,0,0.0001,0.001)
        # self.measure_x_p=kf_lib.KF_updater(1,self.kf_x)
        # self.measure_x_p.constantSpeed_Position(1.0)

        # self.kf_y=kf_lib.KalmanFilter(2)
        # self.kf_y.constantSpeed(dt,0,0,0.0001,0.001)
        # self.measure_y_p=kf_lib.KF_updater(1,self.kf_y)
        # self.measure_y_p.constantSpeed_Position(1.0)

        # self.kf_z=kf_lib.KalmanFilter(2)
        # self.kf_z.constantSpeed(dt,0,0,0.0001,0.001)
        # self.measure_z_p=kf_lib.KF_updater(1,self.kf_z)
        # self.measure_z_p.constantSpeed_Position(1.0)

        # self.kf_th=kf_lib.KalmanFilter(2)
        # self.kf_th.constantSpeed(dt,1.57,0,0.01,0.01)
        # self.measure_th_p=kf_lib.KF_updater(1,self.kf_th)
        # self.measure_th_p.constantSpeed_Position(1)
        # constantSpeed ====================================

        self.after_filter_pub=rospy.Publisher('target'+str(_i)+"_f", Twist, queue_size=1)
        self.kf_pmat_pub=rospy.Publisher('kf_pmat_'+str(_i), Twist, queue_size=1)
        self.t=time.time()
    def cb_fun(self,_data):
        self.measure_x_p.update(_data.linear.x)
        self.measure_y_p.update(_data.linear.y)
        self.measure_z_p.update(_data.linear.z)
        self.measure_th_p.update(_data.angular.z)
        

        self.data=Twist()
        self.data.linear.x=self.kf_x.X[0][0]
        self.data.linear.y=self.kf_y.X[0][0]
        self.data.linear.z=self.kf_z.X[0][0]
        self.data.angular.z=self.kf_th.X[0][0]
        self.after_filter_pub.publish(self.data)

        kf_pmat_pub_msg=Twist()
        kf_pmat_pub_msg.linear.x=self.kf_x.P[0][0]
        kf_pmat_pub_msg.linear.y=self.kf_y.P[0][0]
        kf_pmat_pub_msg.linear.z=self.kf_z.P[0][0]
        kf_pmat_pub_msg.angular.z=self.kf_th.P[0][0]
        self.kf_pmat_pub.publish(kf_pmat_pub_msg)

        self.t=time.time()
    def isTimeout(self):
        if (time.time()-self.t<50) and (not (self.data is None)):
            return 0
        else:
            return 1

rospy.init_node('plan_node', anonymous=True)
plan_wp_pub = rospy.Publisher('plan_wp', Twist, queue_size=1)
target_set=[target(i) for i in range(51,60)]
rate = rospy.Rate(1.0/dt)
vper=viewPanning.viewPanner()
while not rospy.is_shutdown():
    
    for i in [0,1]:
        target_set[i].kf_x.prediction([])
        target_set[i].kf_y.prediction([])
        target_set[i].kf_z.prediction([])
        target_set[i].kf_th.prediction([])
    if target_set[0].isTimeout() and target_set[1].isTimeout() :
        pass
    elif target_set[0].isTimeout() and (not target_set[1].isTimeout()):
        res=target_set[1].data
        plan_wp_pub.publish(res)
    elif (not target_set[0].isTimeout()) and (target_set[1].isTimeout()):
        res=target_set[0].data
        plan_wp_pub.publish(res)
    elif (not target_set[0].isTimeout()) and (not target_set[1].isTimeout()):
        # res=mulitTarget.TwoTargetPos(target_set[0].data,target_set[1].data)
        taskPoint=viewPanning.twist2taskpoint([target_set[0].data,target_set[1].data])
        vper.set_taskPoint(taskPoint)
        res=viewPanning.ci2twist(vper.gant(times=50))
        plan_wp_pub.publish(res)
    rate.sleep()