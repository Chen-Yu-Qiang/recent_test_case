#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
import threading
import time

class box_data:
    def __init__(self):
        self.x=0
        self.y=0
        self.w=0
        self.lock=threading.Lock()
        self.newtime=time.time()
        self.delta=Twist()
    
    def setFromMsg(self,data):
        if data.x * data.y * data.z==0:
            return
        self.lock.acquire()
        self.x = data.x
        self.y = data.y
        self.w = data.z
        self.newtime = time.time()
        self.lock.release()

    def getXYZ(self,pixAT1m):
        self.lock.acquire()
        distance = pixAT1m / (self.w+0.001)
        x_now = min(distance,10)
        y_now = (((self.x-480) * distance) / 952)*(-1)
        z_now = ((self.y-360) * distance) / 952
        self.lock.release()
        box_pub_msg = Twist()
        box_pub_msg.linear.x = x_now
        box_pub_msg.linear.y = y_now
        box_pub_msg.linear.z = z_now+0.9
        return box_pub_msg
    
    def set_delta(self,delta):
        self.delta = delta


    def isTimeOut(self):
        if (time.time()-self.newtime)>0.1:
            return True
        else:
            return False

def get_deltaXYZ(p1,p2):
    delta_msg = Twist()
    
    if p1.isTimeOut() or p2.isTimeOut():
        delta_msg.linear.x = 0
        delta_msg.linear.y = 0
        delta_msg.linear.z = 0
        return delta_msg
    box_msg_p1 = p1.getXYZ(191)
    box_msg_p2 = p2.getXYZ(191)
    delta_msg.linear.x = box_msg_p2.linear.x - box_msg_p1.linear.x
    delta_msg.linear.y = box_msg_p2.linear.y - box_msg_p1.linear.y
    delta_msg.linear.z = box_msg_p2.linear.z - box_msg_p1.linear.z
    p1.set_delta(delta_msg)
    return delta_msg



box_data_r=box_data()
box_data_g=box_data()
box_data_b=box_data()


def cb_box_r(data):
    global box_data_r
    box_data_r.setFromMsg(data)

def cb_box_g(data):
    global box_data_g
    box_data_g.setFromMsg(data)

def cb_box_b(data):
    global box_data_b
    box_data_b.setFromMsg(data)


rospy.init_node('link_box_node', anonymous=True)
box_sub_r = rospy.Subscriber('box_in_img_r', Point, cb_box_r)
box_sub_g = rospy.Subscriber('box_in_img_g', Point, cb_box_g)
box_sub_b = rospy.Subscriber('box_in_img_b', Point, cb_box_b)

box_pub_r = rospy.Publisher('from_box_r', Twist, queue_size=1)
box_pub_g = rospy.Publisher('from_box_g', Twist, queue_size=1)
box_pub_b = rospy.Publisher('from_box_b', Twist, queue_size=1)
box_pub_m = rospy.Publisher('from_box_merge', Twist, queue_size=1)
box_pub_r2g = rospy.Publisher('from_box_r2g', Twist, queue_size=1)
box_pub_g2b = rospy.Publisher('from_box_g2b', Twist, queue_size=1)
rate = rospy.Rate(30)


while  not rospy.is_shutdown():

    if not box_data_r.isTimeOut():
        box_pub_r_msg=box_data_r.getXYZ(191)
        box_pub_r.publish(box_pub_r_msg)
        box_pub_m.publish(box_pub_r_msg)
        if not box_data_g.isTimeOut():
            box_pub_r2g.publish(get_deltaXYZ(box_data_r,box_data_g))
    elif not box_data_g.isTimeOut():
        box_pub_g_msg=box_data_g.getXYZ(191)
        box_pub_g.publish(box_pub_g_msg)
        box_pub_m.publish(box_pub_g_msg)
        if not box_data_b.isTimeOut():
            box_pub_g2b.publish(get_deltaXYZ(box_data_g,box_data_b))
    elif not box_data_b.isTimeOut():
        box_pub_b_msg=box_data_b.getXYZ(191)
        box_pub_m.publish(box_pub_b_msg)
        box_pub_b.publish(box_pub_b_msg)

    
    rate.sleep()