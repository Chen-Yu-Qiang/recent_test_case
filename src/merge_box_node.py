#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
import threading
import time
from std_msgs.msg import Float32
import numpy as np
import BoardRanking
from sensor_msgs.msg import Image
from tello_driver.msg import TelloStatus
from cv_bridge import CvBridge, CvBridgeError
import cv2
class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/tello_raw",Image,self.callback)
        self.cv_image=None

    def callback(self,data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        except CvBridgeError as e:
            print(e)


class box_data:
    def __init__(self):
        self.x=0
        self.y=0
        self.w=0
        self.lock=threading.Lock()
        self.newtime=time.time()
    
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
    

    

    def isTimeOut(self):
        if (time.time()-self.newtime)>0.2:
            return True
        else:
            return False

def getInvXYZ(org_msg,img_ang,kf_msg):
    InvXYZ_msg = Twist()
    InvXYZ_msg.linear.x = kf_msg.linear.x-org_msg.linear.x
    InvXYZ_msg.linear.y = kf_msg.linear.y-org_msg.linear.y
    InvXYZ_msg.linear.z = kf_msg.linear.z-org_msg.linear.z
    InvXYZ_msg.angular.z = kf_msg.angular.z-img_ang+np.pi/2
    return InvXYZ_msg

ang=1.571
def cb_ang(data):
    global ang
    ang=-data.data+1.571
def cb_kf_now(data):
    global ang
    ang=-data.angular.z+1.571

def Rz(data):
    global ang
    
    # ang=0
    out_msg=Twist()
    out_msg.linear.x = data.linear.x*(np.cos(ang))+data.linear.y*(np.sin(ang))
    out_msg.linear.y = data.linear.x*(np.sin(-ang))+data.linear.y*(np.cos(ang))
    out_msg.linear.z = data.linear.z 
    # print(data,out_msg)
    return out_msg

m_box=0
kf_data=Twist()
def cb_box(data):
    global ranking_m_pub,kf_data,ang
    ang=-data.angular.z+1.571
    kf_data= data
    m_box=BoardRanking.ranking(data)
    ranking_m_pub.publish(m_box)


box_data_r=box_data()
box_data_g=box_data()
box_data_b=box_data()

from_img_ang=0
def cb_img_ang(data):
    global from_img_ang
    from_img_ang=data.data

def cb_box_r(data):
    global box_data_r
    box_data_r.setFromMsg(data)

def cb_box_g(data):
    global box_data_g
    box_data_g.setFromMsg(data)

def cb_box_b(data):
    global box_data_b
    box_data_b.setFromMsg(data)

ic = image_converter()
rospy.init_node('merge_box_node', anonymous=True)
box_sub_r = rospy.Subscriber('box_in_img_r', Point, cb_box_r)
box_sub_g = rospy.Subscriber('box_in_img_g', Point, cb_box_g)
box_sub_b = rospy.Subscriber('box_in_img_b', Point, cb_box_b)
# ang_sub = rospy.Subscriber('kf_ang', Float32, cb_ang)
# kf_now_sub = rospy.Subscriber('from_kf', Float32, cb_kf_now)
img_ang_sub=rospy.Subscriber('from_img_ang', Float32, cb_img_ang)
box_pub_r = rospy.Publisher('from_box_r', Twist, queue_size=1)
box_pub_g = rospy.Publisher('from_box_g', Twist, queue_size=1)
box_pub_b = rospy.Publisher('from_box_b', Twist, queue_size=1)
box_pub_r_target = rospy.Publisher('from_box_r_target', Twist, queue_size=1)
box_pub_g_target = rospy.Publisher('from_box_g_target', Twist, queue_size=1)
box_pub_b_target = rospy.Publisher('from_box_b_target', Twist, queue_size=1)
box_pub_m_before = rospy.Publisher('from_box_merge_before', Twist, queue_size=1)
box_pub_m_after = rospy.Publisher('from_box_merge', Twist, queue_size=1)
target_pub = rospy.Publisher('target', Twist, queue_size=1)
ranking_m_pub = rospy.Publisher('ranking_m', Float32, queue_size=1)
ranking_aruco_pub = rospy.Publisher('ranking_aruco', Float32, queue_size=1)
img_ang_pub=rospy.Publisher('from_img_ang2', Float32, queue_size=1)
box_sub = rospy.Subscriber('from_kf', Twist, cb_box)
rate = rospy.Rate(30)

x_margin = 50
y_margin = 100
NeedAruco=1
while  not rospy.is_shutdown():


    if box_data_b.isTimeOut() or box_data_g.isTimeOut() or box_data_r.isTimeOut():
        NeedAruco=1
        pass
    elif box_data_r.x< box_data_g.x or box_data_r.x< box_data_b.x or box_data_b.y > box_data_g.y or  box_data_b.y > box_data_r.y:
        pass
    elif box_data_r.x < x_margin or box_data_r.x > 960-x_margin  or box_data_r.y < y_margin or box_data_r.y > 720-y_margin:
        pass
    else:
        if NeedAruco==1:
            res=BoardRanking.checkAruco(ic.cv_image)
            if not res==-1:
                m_box=res
                NeedAruco=0
                ranking_aruco_pub.publish(m_box)
        if m_box<50:
            # the board is for positioning
            box_pub_r_msg=box_data_r.getXYZ(184)
            box_pub_g_msg=box_data_g.getXYZ(184)
            box_pub_b_msg=box_data_b.getXYZ(184)
            box_pub_r.publish(Rz(box_pub_r_msg))
            box_pub_g.publish(Rz(box_pub_g_msg))
            box_pub_b.publish(Rz(box_pub_b_msg))
            box_pub_m_after.publish(BoardRanking.gotoOrg(m_box,Rz(box_pub_r_msg)))
            box_pub_m_before.publish(Rz(box_pub_r_msg))
            img_ang_pub.publish(from_img_ang)
        else:
            # the board is target
            box_pub_r_msg_target=box_data_r.getXYZ(184)
            box_pub_g_msg_target=box_data_g.getXYZ(184)
            box_pub_b_msg_target=box_data_b.getXYZ(184)
            box_pub_r_target.publish(Rz(box_pub_r_msg_target))
            box_pub_g_target.publish(Rz(box_pub_g_msg_target))
            box_pub_b_target.publish(Rz(box_pub_b_msg_target))
            target_pub.publish(getInvXYZ(Rz(box_pub_r_msg_target),from_img_ang,kf_data))

    
    rate.sleep()