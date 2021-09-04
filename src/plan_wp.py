#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import threading
import mulitTarget
import time
import filter_lib
import kf_lib
import viewPanning
import numpy as np
dt=1.0/30.0

class ros_viewPlanner:
    def __init__(self,myname):
        self.str_name=myname
        self.vper=viewPanning.viewPanner()
        self.ci_pub=rospy.Publisher('plan_wp_'+self.str_name, Twist, queue_size=1)
        self.tpk_pub=rospy.Publisher('plan_tpk_'+self.str_name, Float32MultiArray, queue_size=1)
        self.t=None

    def set_taskPoint(self,t):
        self.vper.set_taskPoint(t)

    def pub_ci(self,time=100):
        res=viewPanning.ci2twist(self.vper.gant(times=time))
        self.ci_pub.publish(res)
    
    def pub_ci_start(self,times=100):
        self.t=threading.Thread(target=self.pub_ci,args=(times,))
        self.t.start()

    def pub_ci_end(self):
        self.t.join()
    
    def pub_tpk(self):
        d=self.vper.get_tpk()
        dd=Float32MultiArray(data=d)
        self.tpk_pub.publish(dd)
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

        self.t=time.time()
    def isTimeout(self):
        if (time.time()-self.t<50) and (not (self.data is None)):
            return 0
        else:
            return 1

rospy.init_node('plan_node', anonymous=True)
target_set=[target(i) for i in range(51,60)]
rate = rospy.Rate(1.0/dt)

rvp_52=ros_viewPlanner("52")
rvp_51_52=ros_viewPlanner("51_52")
rvp_0_51=ros_viewPlanner("0_51")
rvp_0_52=ros_viewPlanner("0_52")





ref_board=Twist()
ref_board.linear.x=0
ref_board.linear.y=0
ref_board.linear.z=0.9
ref_board.angular.z=np.pi*0.5

while not rospy.is_shutdown():
    
    for i in [0,1]:
        target_set[i].kf_x.prediction([])
        target_set[i].kf_y.prediction([])
        target_set[i].kf_z.prediction([])
        target_set[i].kf_th.prediction([])
        

    
    if target_set[0].isTimeout() and target_set[1].isTimeout() :
        pass
    elif target_set[0].isTimeout() and (not target_set[1].isTimeout()):
        taskPoint=viewPanning.twist2taskpoint([target_set[1].data,ref_board])
        rvp_0_52.set_taskPoint(taskPoint)
        rvp_0_52.pub_ci()
        rvp_0_52.pub_tpk()

    elif (not target_set[0].isTimeout()) and (target_set[1].isTimeout()):
        taskPoint=viewPanning.twist2taskpoint([target_set[0].data,ref_board])
        rvp_0_51.set_taskPoint(taskPoint)
        rvp_0_51.pub_ci()
        rvp_0_51.pub_tpk()

    elif (not target_set[0].isTimeout()) and (not target_set[1].isTimeout()):
        taskPoint=viewPanning.twist2taskpoint([target_set[0].data,target_set[1].data])
        rvp_51_52.set_taskPoint(taskPoint)
        rvp_51_52.pub_ci_start()

        taskPoint=viewPanning.twist2taskpoint([target_set[1].data])
        rvp_52.set_taskPoint(taskPoint)
        rvp_52.pub_ci_start()

        taskPoint=viewPanning.twist2taskpoint([target_set[0].data,ref_board])
        rvp_0_51.set_taskPoint(taskPoint)
        rvp_0_51.pub_ci_start()

        taskPoint=viewPanning.twist2taskpoint([target_set[1].data,ref_board])
        rvp_0_52.set_taskPoint(taskPoint)
        rvp_0_52.pub_ci_start()




        rvp_51_52.pub_ci_end()
        rvp_51_52.pub_tpk()
        
        rvp_52.pub_ci_end()
        rvp_52.pub_tpk()

        rvp_0_51.pub_ci_end()
        rvp_0_51.pub_tpk()

        rvp_0_52.pub_ci_end()
        rvp_0_52.pub_tpk()



    rate.sleep()