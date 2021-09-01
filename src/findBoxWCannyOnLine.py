#!/usr/bin/env python
import cv2
import os
import numpy as np
import time
import threading
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
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
DEBUG_MODE=findBoxWCanny.DEBUG_MODE
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
        self.pool=ThreadPool()
        # self.pool=mp.Pool()
        self.together_show=None

    def callback(self,data):
        if time.time()-self.ttt<0.01:
            return
        try:
            self.isdoing=1
            ttt=time.time()
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            xyid,ip = mulitTarget.find_aruco_mean(cv_image)
            if xyid==-1 and ip==-1:
                self.MAIN(cv_image,-1,cv_image.copy(),None,None,None,None)
                return

            img_set = mulitTarget.divImg(ip,cv_image)
            self.together_show=cv_image.copy()
            # for i in range(len(xyid)):
                # cv2.imshow(str(xyid[i][2]),img_set[i])
            #    img_set[i]=cv2.cvtColor(img_set[i], cv2.COLOR_BGR2HSV) 

            # multi==================================================
            if not DEBUG_MODE:
                res = self.pool.map(findBoxWCanny.findRGB, img_set)
                for i in range(len(xyid)):
                    r=None
                    g=None
                    b=None
                    ang=None
                    r,g,b,ang=res[i]
                    # print(r,g,b,ang)
                    self.MAIN(img_set[i],xyid[i][2],cv_image.copy(),r,g,b,ang,ip[i],ip[i+1])

            # single===================================================
            if DEBUG_MODE:
                for i in range(len(xyid)):
                    r=None
                    g=None
                    b=None
                    ang=None                
                    r,g,b,ang=findBoxWCanny.findRGB(img_set[i])
                    # print(r,g,b,ang)
                    self.MAIN(img_set[i],xyid[i][2],cv_image.copy(),r,g,b,ang,ip[i],ip[i+1])


            #============================================================
            # print(time.time()-ttt)
            # print("-------------")
            self.t_show()
        except CvBridgeError as e:
            print(e)
    def MAIN(self,cv_image0,aruco_id,cv_image_org,r,g,b,ang,ipLine1=0,ipLine2=959):
        

        cv_image=self.together_show
        # print(r,g,b,ang)
        if not ang is None:
            if ang<2.618 and ang>0.524:
                pub_ang.publish(ang)
                pp=Twist()
                pp.linear.z=aruco_id
                pp.angular.z=ang
                pub_ang_n.publish(pp)
        if not r is None and not ang is None:
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
            
        if not g is None and not ang is None:
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


        cv_image = cv2.line(cv_image,(int(ipLine1),0),(int(ipLine1),719), (0,0,0), 2)
        cv_image = cv2.line(cv_image,(int(ipLine2),0),(int(ipLine2),719), (0,0,0), 2)
        self.together_show=cv_image

    def t_show(self):
        cv_image=self.together_show
        addbox=cv2.circle(cv_image, (480,360), 5, 255)
        imgAndState = np.hstack((addbox,self.bgimg))
        # imgAndState = cv2.line(imgAndState,(480,0),(480,719), (0,0,0), 1)
        # imgAndState = cv2.line(imgAndState,(0,360),(959,360), (0,0,0), 2)
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
        cv2.waitKey(1)
        self.isdoing=0
        self.ttt=time.time()
        


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
