#!/usr/bin/env python
import os
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import time
from std_msgs.msg import Float32
import kf_lib
def cb_box(data):
    global measure_x_p,measure_y_p,measure_z_p
    ang=12*np.pi/180
    if abs(data.linear.z)>5 or abs(data.linear.y)>15 or abs(data.linear.x)>5 :
        return
    box_x=data.linear.x*np.cos(ang) - data.linear.z*np.sin(ang)
    box_y=data.linear.y
    box_z=data.linear.x*np.sin(ang) + data.linear.z*np.cos(ang)
    

    measure_x_p.update([[box_x]])
    measure_y_p.update([[box_y]])
    measure_z_p.update([[box_z]])

def cb_odom(data):
    global measure_x_v,measure_y_v,measure_z_v
    ang=12*np.pi/180
    vx = -data.twist.twist.angular.x
    vy = data.twist.twist.angular.y
    vz = -data.twist.twist.angular.z
    
    measure_x_v.update([[vx]])
    measure_y_v.update([[vy]])
    measure_z_v.update([[vz]])
    cb_ang_imu(data)

class ang_cont:
    def __init__(self,t0=0):
        self.r=0
        self.unitCircle=np.pi*2
        self.theta=t0
    
    def update(self,t):
        if t-self.theta>0.9*self.unitCircle:
            self.r=self.r-1
        if t-self.theta<(-0.9)*self.unitCircle:
            self.r=self.r+1
        self.theta=t
        return t+self.r*self.unitCircle


ang_imu_obj=ang_cont()
def cb_ang_img(data):
    global measure_th_vp,ang_imu_obj
    ang=data.data+ang_imu_obj.r*ang_imu_obj.unitCircle
    
    measure_th_vp.update([[ang]])

def cb_ang_imu(data):
    global measure_th_imu,ang_imu_obj
    ang=-2*np.arctan2(data.pose.pose.orientation.z,data.pose.pose.orientation.w)


    ang=ang_imu_obj.update(ang)
    measure_th_imu.update([[ang]])


dt=1.0/31

kf_x=kf_lib.KalmanFilter(3)
kf_x.constantSpeedWDrift(dt,1.5,0,0,0.01,0.01,0.000001)
measure_x_p=kf_lib.KF_updater(1,kf_x)
measure_x_p.constantSpeedWDrift_Postition(1)
measure_x_v=kf_lib.KF_updater(1,kf_x)
measure_x_v.constantSpeedWDrift_Speed(1)

kf_y=kf_lib.KalmanFilter(3)
kf_y.constantSpeedWDrift(dt,0,0,0,0.01,0.01,0.000001)
measure_y_p=kf_lib.KF_updater(1,kf_y)
measure_y_p.constantSpeedWDrift_Postition(1)
measure_y_v=kf_lib.KF_updater(1,kf_y)
measure_y_v.constantSpeedWDrift_Speed(1)

kf_z=kf_lib.KalmanFilter(3)
kf_z.constantSpeedWDrift(dt,0,0,0,0.01,0.01,0.000001)
measure_z_p=kf_lib.KF_updater(1,kf_z)
measure_z_p.constantSpeedWDrift_Postition(1)
measure_z_v=kf_lib.KF_updater(1,kf_z)
measure_z_v.constantSpeedWDrift_Speed(1)

kf_th=kf_lib.KalmanFilter(2)
measure_th_vp=kf_lib.KF_updater(1,kf_th)
measure_th_imu=kf_lib.KF_updater(1,kf_th)
kf_th.Q[0][0]=0.01
kf_th.Q[1][1]=0.00000001
kf_th.X[0][0]=np.pi/2
kf_th.X[1][0]=np.pi/2
measure_th_vp.H=np.array([[1,0]])
measure_th_imu.H=np.array([[1,-1]])


rospy.init_node('kf', anonymous=True)
odom_sub = rospy.Subscriber("tello/odom", Odometry, cb_odom)
box_sub = rospy.Subscriber('from_box_merge', Twist, cb_box)
ang_sub = rospy.Subscriber('from_img_ang2', Float32, cb_ang_img)

# imu_sub = rospy.Subscriber('tello/imu', Imu, cb_imu)
# cmd_sub = rospy.Subscriber('tello/cmd_vel', Twist, cb_cmd)
kf_p_pub = rospy.Publisher('from_kf', Twist, queue_size=1)
kf_pmat_pub = rospy.Publisher('kf_pmat', Twist, queue_size=1)
kf_vmean_pub = rospy.Publisher('kf_vmean', Twist, queue_size=1)
kf_p_predict_pub = rospy.Publisher('kf_p_predict', Twist, queue_size=1)
kf_p_measure_pub = rospy.Publisher('kf_p_measure', Twist, queue_size=1)
kf_v_pub = rospy.Publisher('v_kf', Twist, queue_size=1)
kf_ang_pub = rospy.Publisher('kf_ang', Float32 , queue_size=1)
cal_time_pub = rospy.Publisher('cal_time', Float32 , queue_size=1)

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
    kf_pmat_pub.publish(kf_pmat_pub_msg)



    kf_p_msg=Twist()
    kf_p_msg.linear.x=kf_x.X[0][0]
    kf_p_msg.linear.y=kf_y.X[0][0]
    kf_p_msg.linear.z=kf_z.X[0][0]
    kf_p_msg.angular.z=kf_th.X[0][0]
    kf_p_pub.publish(kf_p_msg)
    if kf_x.P[0][0]>0.15:
        kf_p_predict_pub.publish(kf_p_msg)
    else:
        kf_p_measure_pub.publish(kf_p_msg)


    kf_v_msg=Twist()
    kf_v_msg.linear.x=kf_x.X[1][0]
    kf_v_msg.linear.y=kf_y.X[1][0]
    kf_v_msg.linear.z=kf_z.X[1][0]
    kf_v_pub.publish(kf_v_msg)

    kf_vmean_msg=Twist()
    kf_vmean_msg.linear.x=kf_x.X[2][0]
    kf_vmean_msg.linear.y=kf_y.X[2][0]
    kf_vmean_msg.linear.z=kf_z.X[2][0]
    kf_vmean_msg.angular.z=kf_th.X[1][0]
    kf_vmean_pub.publish(kf_vmean_msg)
    
    kf_ang_pub.publish(kf_th.X[0][0])

    
    rate.sleep()