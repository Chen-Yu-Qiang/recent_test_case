#!/usr/bin/env python
import os
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import time
from std_msgs.msg import Float32

def cb_box(data):
    global P,X
    ang=12*np.pi/180
    if abs(data.linear.z)>5 or abs(data.linear.y)>5 or abs(data.linear.x)>5 :
        return
    box_x=data.linear.x*np.cos(ang) - data.linear.z*np.sin(ang)
    box_y=data.linear.y
    box_z=data.linear.x*np.sin(ang) + data.linear.z*np.cos(ang)
    
    Z=np.array([[box_x],[box_y],[box_z]])
    Y=Z-np.dot(H1,X)
    S=np.dot(np.dot(H1,P),np.transpose(H1))+R1
    K=np.dot(np.dot(P,np.transpose(H1)),np.linalg.inv(S))
    
    X=X+np.dot(K,Y)
    P=np.dot(np.eye(9)-np.dot(K,H1),P)

def cb_odom(data):
    global P,X
    ang=12*np.pi/180
    vx = -data.twist.twist.angular.x
    vy = data.twist.twist.angular.y
    vz = -data.twist.twist.angular.z
    
    Z=np.array([[vx],[vy],[vz]])
    Y=Z-np.dot(H2,X)
    S=np.dot(np.dot(H2,P),np.transpose(H2))+R2
    K=np.dot(np.dot(P,np.transpose(H2)),np.linalg.inv(S))
    X=X+np.dot(K,Y)
    P=np.dot(np.eye(9)-np.dot(K,H2),P)
last_time=time.time()
last_data=Twist()
last_data.linear.x=0
last_data.linear.y=0
last_data.linear.z=0
ax=0
ay=0
az=0
def cb_cmd(data):
    global ax,ay,az,last_data,last_time
    return
    delta_t=time.time()-last_time
    ax=(data.linear.x-last_data.linear.x)/delta_t
    ay=(data.linear.y-last_data.linear.y)/delta_t
    az=(data.linear.z-last_data.linear.z)/delta_t
    last_data=data
    last_time=time.time()
def cb_imu(data):
    global ax,ay,az,last_data,last_time
    delta_t=time.time()-last_time
    ax=data.linear_acceleration.x*10
    ay=data.linear_acceleration.y*10
    az=data.linear_acceleration.z*10+9.8
    last_data=data
    last_time=time.time()

dt=1.0/31
# F =np.array(
#    [[1,dt,0,0,0,0],
#     [0,1,0,0,0,0],
#     [0,0,1,dt,0,0],
#     [0,0,0,1,0,0],
#     [0,0,0,0,1,dt],
#     [0,0,0,0,0,1]])

# H1=np.array(
#    [[1,0,0,0,0,0],
#     [0,0,1,0,0,0],
#     [0,0,0,0,1,0]])
# H2=np.array(
#    [[0,1,0,0,0,0],
#     [0,0,0,1,0,0],
#     [0,0,0,0,0,1]])

# B=np.array(
#    [[0.5*dt*dt,0,0],
#     [dt,0,0],
#     [0,0.5*dt*dt,0],
#     [0,dt,0],
#     [0,0,0.5*dt*dt],
#     [0,0,dt]])

# Q = np.eye(6)*0.01
# R1 = np.eye(3)
# R2 = np.eye(3) 
# P = np.eye(6) 
# X=np.zeros((6,1))


F =np.array(
   [[1,dt,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,0,1,dt,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,dt,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,1]])

H1=np.array(
   [[1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0]])
H2=np.array(
   [[0,1,0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0,1,0],
    [0,0,0,0,0,1,0,0,1]])

B=np.array(
   [[0.5*dt*dt,0,0],
    [dt,0,0],
    [0,0.5*dt*dt,0],
    [0,dt,0],
    [0,0,0.5*dt*dt],
    [0,0,dt],
    [0,0,0],
    [0,0,0],
    [0,0,0]])

Q = np.eye(9)*0.01
Q[6][6]=0.000001
Q[7][7]=0.000001
Q[8][8]=0.000001
R1 = np.eye(3)
R2 = np.eye(3) 
P = np.eye(9) 
X=np.zeros((9,1))
X[0]=1.5
rospy.init_node('kf', anonymous=True)
odom_sub = rospy.Subscriber("tello/odom", Odometry, cb_odom)
box_sub = rospy.Subscriber('from_box_merge', Twist, cb_box)

# imu_sub = rospy.Subscriber('tello/imu', Imu, cb_imu)
cmd_sub = rospy.Subscriber('tello/cmd_vel', Twist, cb_cmd)
kf_p_pub = rospy.Publisher('from_kf', Twist, queue_size=1)
kf_pmat_pub = rospy.Publisher('kf_pmat', Twist, queue_size=1)
kf_vmean_pub = rospy.Publisher('kf_vmean', Twist, queue_size=1)
kf_v_pub = rospy.Publisher('v_kf', Twist, queue_size=1)
cal_time_pub = rospy.Publisher('cal_time', Float32 , queue_size=1)

rate = rospy.Rate(1.0/dt)
while  not rospy.is_shutdown():
    #t=time.time()
    U=np.array([[ax],[ay],[az]])
    X = np.dot(F,X)+np.dot(B,U)
    P = np.dot(np.dot(F,P),np.transpose(F)) + Q

    kf_pmat_pub_msg=Twist()
    kf_pmat_pub_msg.linear.x=P[0][0]
    kf_pmat_pub_msg.linear.y=P[2][2]
    kf_pmat_pub_msg.linear.z=P[4][4]
    kf_pmat_pub.publish(kf_pmat_pub_msg)


    kf_p_msg=Twist()
    kf_p_msg.linear.x=X[0]
    kf_p_msg.linear.y=X[2]
    kf_p_msg.linear.z=X[4]
    kf_p_pub.publish(kf_p_msg)
    kf_v_msg=Twist()
    kf_v_msg.linear.x=X[1]
    kf_v_msg.linear.y=X[3]
    kf_v_msg.linear.z=X[5]
    kf_v_pub.publish(kf_v_msg)

    kf_vmean_msg=Twist()
    kf_vmean_msg.linear.x=X[6]
    kf_vmean_msg.linear.y=X[7]
    kf_vmean_msg.linear.z=X[8]
    kf_vmean_pub.publish(kf_vmean_msg)
    
    
    #cal_time_msg=Float32()
    #cal_time_msg.data=time.time()-t
    #cal_time_pub.publish(cal_time_msg)
    
    rate.sleep()