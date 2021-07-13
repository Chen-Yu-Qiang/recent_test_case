#!/usr/bin/env python
import os
import rospy
import numpy as np
from geometry_msgs.msg import Twist
import time
import kf_lib
def cb_box(data):
    global measure_x_p,measure_y_p,measure_z_p,measure_th_p

    box_x=data.linear.x
    box_y=data.linear.y
    box_z=data.linear.z
    box_th=data.angular.z
    

    measure_x_p.update([[box_x]])
    measure_y_p.update([[box_y]])
    measure_z_p.update([[box_z]])
    measure_th_p.update([[box_th]])


dt=1.0/30

# kf_x=kf_lib.KalmanFilter(2)
# kf_x.constantSpeed(dt,2,0,0.01,0.0001)
# measure_x_p=kf_lib.KF_updater(1,kf_x)
# measure_x_p.constantSpeed_Position(1)

# kf_y=kf_lib.KalmanFilter(2)
# kf_y.constantSpeed(dt,-2,0,0.01,0.0001)
# measure_y_p=kf_lib.KF_updater(1,kf_y)
# measure_y_p.constantSpeed_Position(1)

# kf_z=kf_lib.KalmanFilter(2)
# kf_z.constantSpeed(dt,0.4,0,0.0001,0.0001)
# measure_z_p=kf_lib.KF_updater(1,kf_z)
# measure_z_p.constantSpeed_Position(1)


kf_x=kf_lib.KalmanFilter(1)
kf_x.constantPosition(dt,2,0.01)
measure_x_p=kf_lib.KF_updater(1,kf_x)
measure_x_p.constantPosition_Position(1)

kf_y=kf_lib.KalmanFilter(1)
kf_y.constantPosition(dt,-2,0.01)
measure_y_p=kf_lib.KF_updater(1,kf_y)
measure_y_p.constantPosition_Position(1)

kf_z=kf_lib.KalmanFilter(1)
kf_z.constantPosition(dt,0.2,0.01)
measure_z_p=kf_lib.KF_updater(1,kf_z)
measure_z_p.constantPosition_Position(1)

kf_th=kf_lib.KalmanFilter(1)
kf_th.constantPosition(dt,np.pi,0.01)
measure_th_p=kf_lib.KF_updater(1,kf_th)
measure_th_p.constantPosition_Position(1)


rospy.init_node('Target_kf', anonymous=True)
box_sub = rospy.Subscriber('target', Twist, cb_box)
kf_p_pub = rospy.Publisher('target_kf', Twist, queue_size=1)
kf_v_pub = rospy.Publisher('v_target', Twist, queue_size=1)

rate = rospy.Rate(1.0/dt)
while  not rospy.is_shutdown():
    #t=time.time()

    kf_x.prediction([])
    kf_y.prediction([])
    kf_z.prediction([])
    kf_th.prediction([])


    kf_pmat_pub_msg=Twist()
    kf_pmat_pub_msg.linear.x=kf_x.P[0][0]
    kf_pmat_pub_msg.linear.y=kf_y.P[0][0]
    kf_pmat_pub_msg.linear.z=kf_z.P[0][0]
    kf_pmat_pub_msg.angular.z=kf_th.P[0][0]
    # kf_pmat_pub.publish(kf_pmat_pub_msg)



    kf_p_msg=Twist()
    kf_p_msg.linear.x=kf_x.X[0][0]
    kf_p_msg.linear.y=kf_y.X[0][0]
    # kf_p_msg.linear.z=kf_z.X[0][0]
    kf_p_msg.linear.z=0.2
    kf_p_msg.angular.z=kf_th.X[0][0]
    kf_p_pub.publish(kf_p_msg)
    # if kf_x.P[0][0]>0.15:
    #     kf_p_predict_pub.publish(kf_p_msg)
    # else:
    #     kf_p_measure_pub.publish(kf_p_msg)


    # kf_v_msg=Twist()
    # kf_v_msg.linear.x=kf_x.X[1][0]
    # kf_v_msg.linear.y=kf_y.X[1][0]
    # kf_v_msg.linear.z=kf_z.X[1][0]
    # kf_v_pub.publish(kf_v_msg)


    
    # kf_ang_pub.publish(kf_th.X[0][0])

    
    rate.sleep()