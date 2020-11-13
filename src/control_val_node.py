#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from control_msgs.msg import PidState
import threading
import time
box_x=0
box_y=0
box_w=1
box_newTime=time.time()
is_takeoff=1
def cb_box(data):
    global box_lock,box_x,box_y,box_w,box_newTime
    
    if data.x * data.y * data.z==0:
        return
    box_lock.acquire()
    box_x=data.x
    box_y=data.y
    box_w=data.z
    box_newTime=time.time()
    box_lock.release()

def cb_takeoff(data):
    global is_takeoff
    is_takeoff=1

def cb_land(data):
    global is_takeoff
    is_takeoff=0   

def cb_ref(data):
    global ref_lock,x_d,y_d,z_d
    ref_lock.acquire()
    x_d=data.linear.x
    y_d=data.linear.y
    z_d=data.linear.z
    ref_lock.release()



rospy.init_node('control_val_node', anonymous=True)
box_sub = rospy.Subscriber('box_in_img', Point, cb_box)
cmd_val_pub = rospy.Publisher('tello/cmd_vel', Twist, queue_size=1)
x_pid_pub = rospy.Publisher('x_pid', PidState, queue_size=1)
y_pid_pub = rospy.Publisher('y_pid', PidState, queue_size=1)
z_pid_pub = rospy.Publisher('z_pid', PidState, queue_size=1)
box_pub = rospy.Publisher('from_box', Twist, queue_size=1)
takeoff_sub = rospy.Subscriber('tello/takeoff', Empty, cb_takeoff)
ref_sub = rospy.Subscriber('ref', Twist, cb_ref)
land_sub = rospy.Subscriber('tello/land', Empty, cb_land)
rate = rospy.Rate(20)

box_lock=threading.Lock()
ref_lock=threading.Lock()
is_takeoff=1

cmd_val_pub_msg=Twist()
x_d = 1
y_d = 0
z_d = 0
err_x_last=0
err_y_last=0
err_z_last=0
err_x_int=0
err_y_int=0
err_z_int=0
while  not rospy.is_shutdown():
    if is_takeoff:

        d_t = 1

        box_lock.acquire()
        distance = 200 / (box_w+0.001)
        x_now = distance
        y_now = (((box_x-480) * distance) / 952)*(-1)
        z_now = ((box_y-360) * distance) / 952
        box_lock.release()


        x_pid=PidState()
        y_pid=PidState()
        z_pid=PidState()


        ref_lock.acquire()
        err_x = x_d - x_now
        err_y = y_d - y_now
        err_z = z_d - z_now
        ref_lock.release()        




        err_x_dif = (err_x - err_x_last) / d_t
        err_x_int = err_x_int + err_x * d_t
        err_x_last = err_x

        
        err_y_dif = (err_y - err_y_last) / d_t
        err_y_int = err_y_int + err_y * d_t
        err_y_last = err_y

        
        err_z_dif = (err_z - err_z_last) / d_t
        err_z_int = err_z_int + err_z * d_t
        err_z_last = err_z
        
        kp = 0.5
        ki = 0
        kd = 0
        cmd_x=kp*err_x+ki*err_x_int+kd*err_x_dif
        x_pid.error=err_x
        x_pid.p_error=err_x
        x_pid.i_error=err_x_int
        x_pid.d_error=err_x_dif
        x_pid.p_term=kp
        x_pid.i_term=ki
        x_pid.d_term=kd
        x_pid.output=cmd_x

        kp = 0.5
        ki = 0
        kd = 0
        cmd_y=kp*err_y+ki*err_y_int+kd*err_y_dif
        y_pid.error=err_y
        y_pid.p_error=err_y
        y_pid.i_error=err_y_int
        y_pid.d_error=err_y_dif
        y_pid.p_term=kp
        y_pid.i_term=ki
        y_pid.d_term=kd
        y_pid.output=cmd_y


        kp = 0.5
        ki = 0
        kd = 0
        cmd_z=kp*err_z+ki*err_z_int+kd*err_z_dif
        z_pid.error=err_z
        z_pid.p_error=err_z
        z_pid.i_error=err_z_int
        z_pid.d_error=err_z_dif
        z_pid.p_term=kp
        z_pid.i_term=ki
        z_pid.d_term=kd
        z_pid.output=cmd_z





        cmd_val_pub_msg.linear.x = cmd_y
        cmd_val_pub_msg.linear.y = -cmd_x
        cmd_val_pub_msg.linear.z = cmd_z


        if abs(cmd_val_pub_msg.linear.x)>2:
            cmd_val_pub_msg.linear.x=cmd_val_pub_msg.linear.x/abs(cmd_val_pub_msg.linear.x)*2
        if abs(cmd_val_pub_msg.linear.y)>2:
            cmd_val_pub_msg.linear.y=cmd_val_pub_msg.linear.y/abs(cmd_val_pub_msg.linear.y)*2
        if abs(cmd_val_pub_msg.linear.z)>2:
            cmd_val_pub_msg.linear.z=cmd_val_pub_msg.linear.z/abs(cmd_val_pub_msg.linear.z)*2
        
        print(cmd_val_pub_msg)

        
        cmd_val_pub.publish(cmd_val_pub_msg)

        box_pub_msg=Twist()
        box_pub_msg.linear.x = x_now
        box_pub_msg.linear.y = y_now
        box_pub_msg.linear.z = z_now
        box_pub.publish(box_pub_msg)


        x_pid_pub.publish(x_pid)
        y_pid_pub.publish(y_pid)
        z_pid_pub.publish(z_pid)

    rate.sleep()