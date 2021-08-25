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
import filter_lib
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
        self.h=0
        self.aruco_id=-1
        self.area=0
        self.lock=threading.Lock()
        self.newtime=time.time()
    
    def setFromPointMsg(self,data):
        if data.x * data.y * data.z==0:
            return
        self.lock.acquire()
        self.x = data.x
        self.y = data.y
        self.w = data.z
        self.newtime = time.time()
        self.lock.release()
    def setFromTwistMsg(self,data):
        if data.linear.x * data.linear.y * data.angular.x==0:
            return
        self.lock.acquire()
        self.x = data.linear.x
        self.y = data.linear.y
        self.h=data.angular.y
        self.aruco_id=data.linear.z
        self.area=data.angular.z
        self.w = data.angular.x
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
        # box_pub_msg.linear.x = x_now
        # box_pub_msg.linear.y = y_now
        # box_pub_msg.linear.z = z_now+0.9
        ang=12*np.pi/180
        box_pub_msg.linear.x=x_now*np.cos(ang) - z_now*np.sin(ang)
        box_pub_msg.linear.y=y_now
        box_pub_msg.linear.z=x_now*np.sin(ang) + z_now*np.cos(ang)+0.9
        return box_pub_msg
    
    def isTimeOut(self):
        if (time.time()-self.newtime)>0.2:
            return True
        else:
            return False

class board_data:
    def __init__(self,_ArucoID):
        self.r=box_data()
        self.g=box_data()
        self.b=box_data()
        self.img_ang=np.pi/2
        self.ArucoID=_ArucoID
        self.target_filter=filter_lib.meanFilter(3)
        if self.ArucoID>=50:
            self.pub=rospy.Publisher('target'+str(self.ArucoID), Twist, queue_size=1)

    def for_positioning(self):
        box_pub_r_msg=self.r.getXYZ(184)
        box_pub_g_msg=self.g.getXYZ(184)
        box_pub_b_msg=self.b.getXYZ(184)
        box_pub_r.publish(Rz(box_pub_r_msg))
        box_pub_g.publish(Rz(box_pub_g_msg))
        box_pub_b.publish(Rz(box_pub_b_msg))
        box_pub_m_after.publish(BoardRanking.gotoOrg(m_box,Rz(box_pub_r_msg)))
        box_pub_m_before.publish(Rz(box_pub_r_msg))
        img_ang_pub.publish(self.img_ang)

    def for_target(self):
        box_pub_r_msg_target=self.TwistAddArucoID(self.r.getXYZ(184))
        box_pub_r_msg_target=self.target_filter.update(box_pub_r_msg_target)
        box_pub_r_msg_target.linear.z=box_pub_r_msg_target.linear.z-0.9
        box_pub_g_msg_target=self.TwistAddArucoID(self.g.getXYZ(184))
        box_pub_b_msg_target=self.TwistAddArucoID(self.b.getXYZ(184))
        box_pub_r_target.publish(Rz(box_pub_r_msg_target))
        box_pub_g_target.publish(Rz(box_pub_g_msg_target))
        box_pub_b_target.publish(Rz(box_pub_b_msg_target))
        target_pub.publish(getInvXYZ(Rz(box_pub_r_msg_target),self.img_ang,kf_data))
        self.pub.publish(getInvXYZ(Rz(box_pub_r_msg_target),self.img_ang,kf_data))

    def isTimsOut(self):
        if self.b.isTimeOut() or self.g.isTimeOut() or self.r.isTimeOut():
            return 1
        return 0

    def isOutOfBound(self):
        x_margin = 50
        y_margin = 100
        if self.r.x < x_margin or self.r.x > 960-x_margin  or self.r.y < y_margin or self.r.y > 720-y_margin:
            return 1
        if self.g.x < x_margin or self.g.x > 960-x_margin  or self.g.y < y_margin or self.g.y > 720-y_margin:
            return 1
        return 0

    def isErrorOrder(self):
        if self.r.x< self.g.x or self.r.x< self.b.x or self.b.y > self.g.y or  self.b.y > self.r.y:
            return 1
        return 0
    
    def TwistAddArucoID(self,in_msg):
        out_msg=Twist()
        out_msg.linear=in_msg.linear
        if not self.ArucoID==-1:
            out_msg.angular.x=self.ArucoID
            return out_msg
        else:
            print("Aruco ID not get")
            return in_msg

def getInvXYZ(org_msg,img_ang,kf_msg):
    InvXYZ_msg = Twist()
    InvXYZ_msg.linear.x = kf_msg.linear.x-org_msg.linear.x
    InvXYZ_msg.linear.y = kf_msg.linear.y-org_msg.linear.y
    InvXYZ_msg.linear.z = kf_msg.linear.z-org_msg.linear.z
    InvXYZ_msg.angular.z = kf_msg.angular.z-img_ang+np.pi/2
    InvXYZ_msg.angular.x = org_msg.angular.x
    return InvXYZ_msg

ang=1.571


# def cb_ang(data):
#     # no use function
#     global ang
#     ang=-data.data+1.571

# def cb_kf_now(data):
#     # no use function
#     global ang
#     ang=-data.angular.z+1.571

def Rz(data):
    global ang
    
    # ang=0
    out_msg=Twist()
    out_msg.linear.x = data.linear.x*(np.cos(ang))+data.linear.y*(np.sin(ang))
    out_msg.linear.y = data.linear.x*(np.sin(-ang))+data.linear.y*(np.cos(ang))
    out_msg.linear.z = data.linear.z 
    out_msg.angular.x = data.angular.x
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


# box_data_r=box_data()
# box_data_g=box_data()
# box_data_b=box_data()

# from_img_ang=0

board_set=[board_data(i) for i in range(60)]



def cb_img_ang(data):
    global board_set
    board_set[int(data.linear.z)].img_ang=data.angular.z

def cb_box_r(data):
    global board_set
    data.angular.x=data.angular.x/np.sin(board_set[int(data.linear.z)].img_ang)
    print(data.linear.z,data.angular.x,board_set[int(data.linear.z)].img_ang*57.3)
    board_set[int(data.linear.z)].r.setFromTwistMsg(data)

def cb_box_g(data):
    global board_set
    board_set[int(data.linear.z)].g.setFromTwistMsg(data)

def cb_box_b(data):
    global board_set
    board_set[int(data.linear.z)].b.setFromTwistMsg(data)

# ic = image_converter()
rospy.init_node('merge_box_node', anonymous=True)
box_sub_r = rospy.Subscriber('box_in_img_r_n', Twist, cb_box_r)
box_sub_g = rospy.Subscriber('box_in_img_g_n', Twist, cb_box_g)
box_sub_b = rospy.Subscriber('box_in_img_b_n', Twist, cb_box_b)
# ang_sub = rospy.Subscriber('kf_ang', Float32, cb_ang)
# kf_now_sub = rospy.Subscriber('from_kf', Float32, cb_kf_now)
# img_ang_sub=rospy.Subscriber('from_img_ang', Float32, cb_img_ang)
img_ang_sub=rospy.Subscriber('from_img_ang_n', Twist, cb_img_ang)
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


# target_filter=filter_lib.meanFilter(3)

while  not rospy.is_shutdown():

    
    for i in [0]:
        if board_set[i].isTimsOut() or board_set[i].isOutOfBound() or board_set[i].isErrorOrder():
            pass
        else:
            board_set[i].for_positioning()
        
    for i in [51,52]:
        if board_set[i].isTimsOut() or board_set[i].isOutOfBound() or board_set[i].isErrorOrder():
            pass
        else:
            board_set[i].for_target()
    rate.sleep()