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
        self.delta.linear.x=0
        self.delta.linear.y=0
        self.delta.linear.z=0
        self.pixAT1m_=200

    
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
        pixAT1m=self.pixAT1m_
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
    def getXYZDelta(self,pixAT1m=200):
        pixAT1m=self.pixAT1m_
        xyz = self.getXYZ(pixAT1m)
        xyzD=Twist()
        xyzD.linear.x = xyz.linear.x + self.delta.linear.x
        xyzD.linear.y = xyz.linear.y + self.delta.linear.y
        xyzD.linear.z = xyz.linear.z + self.delta.linear.z
        return xyzD

    def set_delta(self,delta):
        self.delta = delta


    def isTimeOut(self):
        if (time.time()-self.newtime)>0.05:
            return True
        else:
            return False

def get_deltaXYZ(p1,p2):
    delta_msg = Twist()
    x_margin = 60
    y_margin = 60
    if p1.isTimeOut() or p2.isTimeOut() or p1.x>960-x_margin or p1.x<x_margin or p1.y>720-y_margin or p1.y<y_margin:
        delta_msg.linear.x = 0
        delta_msg.linear.y = 0
        delta_msg.linear.z = 0
        return delta_msg
    box_msg_p1 = p1.getXYZ(200)
    box_msg_p2 = p2.getXYZ(200)
    delta_msg.linear.x = box_msg_p1.linear.x - box_msg_p2.linear.x+p1.delta.linear.x
    delta_msg.linear.y = box_msg_p1.linear.y - box_msg_p2.linear.y+p1.delta.linear.y
    delta_msg.linear.z = box_msg_p1.linear.z - box_msg_p2.linear.z+p1.delta.linear.z
    p2.set_delta(delta_msg)
    return delta_msg



box_data_r=box_data()
box_data_g=box_data()
box_data_b=box_data()
box_data_r.pixAT1m_=193
box_data_g.pixAT1m_=172



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
box_pub_r2b = rospy.Publisher('from_box_r2b', Twist, queue_size=1)
rate = rospy.Rate(30)


while  not rospy.is_shutdown():
    if not box_data_r.isTimeOut():
        box_pub_r_msg=box_data_r.getXYZ(193)
        box_pub_r.publish(box_pub_r_msg)
    if not box_data_g.isTimeOut():
        box_pub_g_msg=box_data_g.getXYZ(172)
        box_pub_g.publish(box_pub_g_msg)
    if not box_data_b.isTimeOut():
        box_pub_b_msg=box_data_b.getXYZ(200)
        box_pub_b.publish(box_pub_b_msg)


    if not box_data_r.isTimeOut():
        box_pub_m.publish(box_data_r.getXYZDelta())
        if (not box_data_g.isTimeOut()) and box_data_r.getXYZ(200).linear.x>1.9:
            r2g_msg=get_deltaXYZ(box_data_r,box_data_g)
            if r2g_msg.linear.x==0 and r2g_msg.linear.y==0 and r2g_msg.linear.z==0:
                pass
            else:
                box_pub_r2g.publish(r2g_msg)
            if not box_data_b.isTimeOut():
                r2b_msg=get_deltaXYZ(box_data_g,box_data_b)
                if r2b_msg.linear.x==0 and r2b_msg.linear.y==0 and r2b_msg.linear.z==0:
                    pass
                else:
                    box_pub_r2b.publish(r2b_msg)
    elif not box_data_g.isTimeOut():
        box_pub_m.publish(box_data_g.getXYZDelta())
        if (not box_data_b.isTimeOut()) and box_data_b.getXYZ(200).linear.x>1.9:
            r2b_msg=get_deltaXYZ(box_data_g,box_data_b)
            if r2b_msg.linear.x==0 and r2b_msg.linear.y==0 and r2b_msg.linear.z==0:
                pass
            else:
                box_pub_r2b.publish(r2b_msg)
    elif not box_data_b.isTimeOut():
        box_pub_m.publish(box_data_b.getXYZDelta())

    
    rate.sleep()