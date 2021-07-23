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
import mulitTarget
box_x=0
box_y=0
box_z=1
box_t=0
x_d = 1
y_d = 0
z_d = 0
power_last=-1
def cb_box(data):
    global box_x,box_y,box_z,box_t
    box_x=data.linear.x
    box_y=data.linear.y
    box_z=data.linear.z
    box_t=data.angular.z

ang_d=1.571
def cb_ref(data):
    global x_d,y_d,z_d,ang_d
    x_d=data.linear.x
    y_d=data.linear.y
    z_d=data.linear.z
    ang_d=data.angular.z

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
        self.ttt=time.time()
    def callback(self,data):
        if time.time()-self.ttt<0.01:
            return
        
        try:
            self.isdoing=1
            ttt=time.time()
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            xyid,ip = mulitTarget.find_aruco_mean(cv_image)
            if xyid==-1 and ip==-1:
                self.MAIN(cv_image,-1,cv_image)
                return
            img_set = mulitTarget.divImg(ip,cv_image)
            for i in range(len(xyid)):
                self.MAIN(img_set[i],xyid[i][2],cv_image.copy())
        except CvBridgeError as e:
            print(e)
    def MAIN(self,cv_image,aruco_id,cv_image_org):
        r,g,b,ang=findBoxWCanny.findRGB(cv_image)
        cv_image=cv_image_org
        if not ang is None:
            if ang<2.618 and ang>0.524:
                pub_ang.publish(ang)
                pp=Twist()
                pp.linear.z=aruco_id
                pp.angular.z=ang
                pub_ang_n.publish(pp)
        if not r is None:
            # print(r)
            x, y, w, h, a =findBoxWCanny.xywh(findBoxWCanny.div1234(r))
            if not x*y*w*h==0:
                p=Point()
                p.x=x
                p.y=y
                p.z=w
                pub_r.publish(p)
                pp=Twist()
                pp.linear.x=x
                pp.linear.y=y
                pp.linear.z=aruco_id
                pp.angular.x=w
                pp.angular.y=h
                pp.angular.z=a
                pub_r_n.publish(pp)
                cv_image=cv2.rectangle(cv_image, (int(x-0.5*w),int(y-0.5*h)), (int(x+0.5*w),int(y+0.5*h)), (0,0,255), 5) 
                cv_image = cv2.putText(cv_image,str(aruco_id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            
        if not g is None:
            x, y, w, h , a= findBoxWCanny.xywh(findBoxWCanny.div1234(g))
            if not x*y*w*h==0:
                p=Point()
                p.x=x
                p.y=y
                p.z=w
                pub_g.publish(p)
                pp=Twist()
                pp.linear.x=x
                pp.linear.y=y
                pp.linear.z=aruco_id
                pp.angular.x=w
                pp.angular.y=h
                pp.angular.z=a
                pub_g_n.publish(pp)
                cv_image=cv2.rectangle(cv_image, (int(x-0.5*w),int(y-0.5*h)), (int(x+0.5*w),int(y+0.5*h)), (0,255,0), 5) 
                cv_image = cv2.putText(cv_image,str(aruco_id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        if not b is None:
            x, y, w, h, a = findBoxWCanny.xywh(findBoxWCanny.div1234(b))
            if not x*y*w*h==0:
                p=Point()
                p.x=x
                p.y=y
                p.z=w
                pub_b.publish(p)
                pp=Twist()
                pp.linear.x=x
                pp.linear.y=y
                pp.linear.z=aruco_id
                pp.angular.x=w
                pp.angular.y=h
                pp.angular.z=a
                pub_b_n.publish(pp)
                cv_image=cv2.rectangle(cv_image, (int(x-0.5*w),int(y-0.5*h)), (int(x+0.5*w),int(y+0.5*h)), (255,0,0), 5) 
                cv_image = cv2.putText(cv_image,str(aruco_id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)


        addbox=cv2.circle(cv_image, (480,360), 5, 255)
        imgAndState = np.hstack((addbox,self.bgimg))
        imgAndState = cv2.line(imgAndState,(480,0),(480,719), (0,0,0), 1)
        imgAndState = cv2.putText(imgAndState,str(power_last),(1130,200),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        imgAndState = cv2.putText(imgAndState,str(box_x),(1130,280),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        imgAndState = cv2.putText(imgAndState,str(box_y),(1130,320),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        imgAndState = cv2.putText(imgAndState,str(box_z),(1130,360),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        imgAndState = cv2.putText(imgAndState,str(box_t*57.296),(1130,400),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        imgAndState = cv2.putText(imgAndState,str(x_d),(1130,435),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        imgAndState = cv2.putText(imgAndState,str(y_d),(1130,475),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        imgAndState = cv2.putText(imgAndState,str(z_d),(1130,515),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        imgAndState = cv2.putText(imgAndState,str(ang_d*57.296),(1130,555),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow('addbox', imgAndState)
        self.isdoing=0
        self.ttt=time.time()
        cv2.waitKey(1)
        


ic = image_converter()
rospy.init_node('image_converter', anonymous=True)
pub_r = rospy.Publisher('box_in_img_r', Point, queue_size=1)
pub_g = rospy.Publisher('box_in_img_g', Point, queue_size=1)
pub_b = rospy.Publisher('box_in_img_b', Point, queue_size=1)
pub_r_n = rospy.Publisher('box_in_img_r_n', Twist, queue_size=1)
pub_g_n = rospy.Publisher('box_in_img_g_n', Twist, queue_size=1)
pub_b_n = rospy.Publisher('box_in_img_b_n', Twist, queue_size=1)
pub_ang=rospy.Publisher('from_img_ang', Float32, queue_size=1)
pub_ang_n=rospy.Publisher('from_img_ang_n', Twist, queue_size=1)
ref_sub = rospy.Subscriber('ref', Twist, cb_ref)
box_sub = rospy.Subscriber('from_kf', Twist, cb_box)
power_sub = rospy.Subscriber('tello/status', TelloStatus, cb_power)
rospy.spin()
