#!/usr/bin/env python
import cv2
import os
import numpy as np
import time
import threading
import rospy
import findBoxWCanny
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from tello_driver.msg import TelloStatus
from cv_bridge import CvBridge, CvBridgeError
import time

box_x=0
box_y=0
box_z=1
x_d = 1
y_d = 0
z_d = 0
power_last=100
def cb_box(data):
    global box_x,box_y,box_z
    box_x=data.linear.x
    box_y=data.linear.y
    box_z=data.linear.z


def cb_ref(data):
    global x_d,y_d,z_d
    x_d=data.linear.x
    y_d=data.linear.y
    z_d=data.linear.z

def cb_power(data):
    global power_last
    power_last=data.battery_percentage


r=None
g=None
b=None


class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/tello_raw",Image,self.callback)
        self.bgimg = cv2.imread("bg.png")
        self.isdoing=0

    def callback(self,data):
        try:
            
            self.isdoing=1
            ttt=time.time()
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            r,g,b,ang=findBoxWCanny.findRGB(cv_image)
            if not ang is None:
                if ang<150 and ang>30:
                    pub_ang.publish(ang)
            if not r is None:
                # print(r)
                x, y, w, h, _ =findBoxWCanny.xywh(findBoxWCanny.div1234(r))
                if not x*y*w*h==0:
                    p=Point()
                    p.x=x
                    p.y=y
                    p.z=w
                    pub_r.publish(p)
                    cv_image=cv2.rectangle(cv_image, (int(x-0.5*w),int(y-0.5*h)), (int(x+0.5*w),int(y+0.5*h)), (0,0,255), 5) 
                
            if not g is None:
                x, y, w, h , _= findBoxWCanny.xywh(findBoxWCanny.div1234(g))
                if not x*y*w*h==0:
                    p=Point()
                    p.x=x
                    p.y=y
                    p.z=w
                    pub_g.publish(p)
                    cv_image=cv2.rectangle(cv_image, (int(x-0.5*w),int(y-0.5*h)), (int(x+0.5*w),int(y+0.5*h)), (0,255,0), 5) 
            if not b is None:
                x, y, w, h, _ = findBoxWCanny.xywh(findBoxWCanny.div1234(b))
                if not x*y*w*h==0:
                    p=Point()
                    p.x=x
                    p.y=y
                    p.z=w
                    pub_b.publish(p)
                    cv_image=cv2.rectangle(cv_image, (int(x-0.5*w),int(y-0.5*h)), (int(x+0.5*w),int(y+0.5*h)), (255,0,0), 5) 


            addbox=cv2.circle(cv_image, (480,360), 5, 255)
            imgAndState = np.hstack((addbox,self.bgimg))
            imgAndState = cv2.line(imgAndState,(480,0),(480,719), (0,0,0), 1)
            imgAndState = cv2.putText(imgAndState,str(power_last),(1130,200),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(box_x),(1130,280),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(box_y),(1130,320),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(box_z),(1130,360),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(ang),(1130,400),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(x_d),(1130,435),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(y_d),(1130,475),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(z_d),(1130,515),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            cv2.imshow('addbox', imgAndState)
            self.isdoing=0
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)


ic = image_converter()
rospy.init_node('image_converter', anonymous=True)
pub_r = rospy.Publisher('box_in_img_r', Point, queue_size=1)
pub_g = rospy.Publisher('box_in_img_g', Point, queue_size=1)
pub_b = rospy.Publisher('box_in_img_b', Point, queue_size=1)
pub_ang=rospy.Publisher('from_img_ang', Float32, queue_size=1)
ref_sub = rospy.Subscriber('ref', Twist, cb_ref)
box_sub = rospy.Subscriber('from_kf', Twist, cb_box)
power_sub = rospy.Subscriber('tello/status', TelloStatus, cb_power)
rospy.spin()

