#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import threading
import mulitTarget
import time
import kf_lib

import numpy as np
dt=1.0/30.0

class target:
    def __init__(self,_i):
        self.sub = rospy.Subscriber('target'+str(_i), Twist, self.cb_fun)
        self.i=_i
        self.data=None
        self.future=Twist()
        # constantPosition ====================================
        # self.kf_x=kf_lib.KalmanFilter(1)
        # self.kf_x.constantPosition(dt,0,0.001)
        # self.measure_x_p=kf_lib.KF_updater(1,self.kf_x)
        # self.measure_x_p.constantPosition_Position(1)

        # self.kf_y=kf_lib.KalmanFilter(1)
        # self.kf_y.constantPosition(dt,0,0.001)
        # self.measure_y_p=kf_lib.KF_updater(1,self.kf_y)
        # self.measure_y_p.constantPosition_Position(1)

        # self.kf_z=kf_lib.KalmanFilter(1)
        # self.kf_z.constantPosition(dt,0,0.001)
        # self.measure_z_p=kf_lib.KF_updater(1,self.kf_z)
        # self.measure_z_p.constantPosition_Position(1)

        # self.kf_th=kf_lib.KalmanFilter(1)
        # self.kf_th.constantPosition(dt,1.57,0.005)
        # self.measure_th_p=kf_lib.KF_updater(1,self.kf_th)
        # self.measure_th_p.constantPosition_Position(1)
        # constantPosition ====================================

        # constantSpeed ====================================
        self.kf_x=kf_lib.KalmanFilter(2)
        self.kf_x.constantSpeed(dt,0,0,0.0001,0.001)
        self.measure_x_p=kf_lib.KF_updater(1,self.kf_x)
        self.measure_x_p.constantSpeed_Position(1.0)

        self.kf_y=kf_lib.KalmanFilter(2)
        self.kf_y.constantSpeed(dt,0,0,0.0001,0.001)
        self.measure_y_p=kf_lib.KF_updater(1,self.kf_y)
        self.measure_y_p.constantSpeed_Position(1.0)

        self.kf_z=kf_lib.KalmanFilter(2)
        self.kf_z.constantSpeed(dt,0,0,0.0001,0.001)
        self.measure_z_p=kf_lib.KF_updater(1,self.kf_z)
        self.measure_z_p.constantSpeed_Position(1.0)

        self.kf_th=kf_lib.KalmanFilter(2)
        self.kf_th.constantSpeed(dt,1.57,0,0.01,0.01)
        self.measure_th_p=kf_lib.KF_updater(1,self.kf_th)
        self.measure_th_p.constantSpeed_Position(1)
        # constantSpeed ====================================

        self.after_filter_pub=rospy.Publisher('target'+str(_i)+"_f", Twist, queue_size=1)
        self.filter_future_pub=rospy.Publisher('target'+str(_i)+"_f_future", Twist, queue_size=1)
        self.kf_pmat_pub=rospy.Publisher('kf_pmat_'+str(_i), Twist, queue_size=1)
        self.t=time.time()
    def cb_fun(self,_data):
        aaa=time.time()
        self.measure_x_p.update(_data.linear.x)
        self.measure_y_p.update(_data.linear.y)
        self.measure_z_p.update(_data.linear.z)
        self.measure_th_p.update(_data.angular.z)

        kf_pmat_pub_msg=Twist()
        kf_pmat_pub_msg.linear.x=self.kf_x.P[0][0]
        kf_pmat_pub_msg.linear.y=self.kf_y.P[0][0]
        kf_pmat_pub_msg.linear.z=self.kf_z.P[0][0]
        kf_pmat_pub_msg.angular.z=self.kf_th.P[0][0]
        self.kf_pmat_pub.publish(kf_pmat_pub_msg)
        
        if kf_pmat_pub_msg.linear.x<0.1:
            self.data=Twist()
            self.data.linear.x=self.kf_x.X[0][0]
            self.data.linear.y=self.kf_y.X[0][0]
            self.data.linear.z=self.kf_z.X[0][0]
            self.data.angular.z=self.kf_th.X[0][0]
            self.after_filter_pub.publish(self.data)


            self.future=Twist()
            self.future.linear.x=self.kf_x.get_future()[0][0]
            self.future.linear.y=self.kf_y.get_future()[0][0]
            self.future.linear.z=self.kf_z.get_future()[0][0]
            self.future.angular.z=self.kf_th.get_future()[0][0]
            self.filter_future_pub.publish(self.future)

        self.t=time.time()
        # print(self.t-aaa)
    def isTimeout(self):
        if (time.time()-self.t<50) and (not (self.data is None)):
            return 0
        else:
            return 1

rospy.init_node('target_kf_node', anonymous=True)
target_set=[target(i) for i in range(51,60)]
rate = rospy.Rate(1.0/dt)


while not rospy.is_shutdown():
    
    for i in [0,1]:
        target_set[i].kf_x.prediction([])
        target_set[i].kf_y.prediction([])
        target_set[i].kf_z.prediction([])
        target_set[i].kf_th.prediction([])
        


    rate.sleep()