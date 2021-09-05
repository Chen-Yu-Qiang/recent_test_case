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
        self.sub = rospy.Subscriber('target'+str(_i)+"_f", Twist, self.cb_fun)
        self.sub2 = rospy.Subscriber('target'+str(_i)+"_f_future", Twist, self.cb_fun2)
        self.i=_i
        self.data=None
        self.future=Twist()
        self.t=time.time()
    def cb_fun(self,_data):
        self.data=_data
        self.t=time.time()
    def cb_fun2(self,_data):
        self.future=_data
    def isTimeout(self):
        if (time.time()-self.t<50) and (not (self.data is None)):
            return 0
        else:
            return 1

rospy.init_node('plan_node', anonymous=True)
target_set=[target(i) for i in range(51,60)]
rate = rospy.Rate(1.0/dt)

rvp_52=ros_viewPlanner("52")
rvp_52_future=ros_viewPlanner("52_future")
rvp_51_52_future=ros_viewPlanner("51_52_future")
rvp_51_52=ros_viewPlanner("51_52")
rvp_0_51=ros_viewPlanner("0_51")
rvp_0_52=ros_viewPlanner("0_52")





ref_board=Twist()
ref_board.linear.x=0
ref_board.linear.y=0
ref_board.linear.z=0.9
ref_board.angular.z=np.pi*0.5

while not rospy.is_shutdown():

    
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

        taskPoint=viewPanning.twist2taskpoint([target_set[0].data,target_set[1].data,target_set[1].future])
        rvp_51_52_future.set_taskPoint(taskPoint)
        rvp_51_52_future.pub_ci_start()

        taskPoint=viewPanning.twist2taskpoint([target_set[1].data,target_set[1].future])
        rvp_52_future.set_taskPoint(taskPoint)
        rvp_52_future.pub_ci_start()



        rvp_51_52.pub_ci_end()
        rvp_51_52.pub_tpk()
        
        rvp_52.pub_ci_end()
        rvp_52.pub_tpk()

        rvp_0_51.pub_ci_end()
        rvp_0_51.pub_tpk()

        rvp_0_52.pub_ci_end()
        rvp_0_52.pub_tpk()

        rvp_51_52_future.pub_ci_end()
        rvp_51_52_future.pub_tpk()
        
        rvp_52_future.pub_ci_end()
        rvp_52_future.pub_tpk()

    rate.sleep()