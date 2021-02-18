#!/usr/bin/env python
import cv2
import os
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
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

def findRect(img,color):
    def nothing(data):
        pass

    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    # print(hsv[360][480])
    if color=="g":
        if time.localtime().tm_hour<=18 and time.localtime().tm_hour>=6 :
            lower_g = np.array([67, 193, 30])
            upper_g = np.array([78, 253, 75])
        else:
            lower_g = np.array([67, 102, 109])
            upper_g = np.array([78, 229, 251])

        mask=cv2.inRange(hsv, lower_g, upper_g)


    if color=="r":

        if time.localtime().tm_hour<=18 and time.localtime().tm_hour>=6 :
            lower_red = np.array([174, 165, 35])
            upper_red = np.array([179, 255, 93])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 165, 35])
            upper_red = np.array([4, 255, 93])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            mask=cv2.bitwise_or(mask1,mask2)
        else:
            lower_red = np.array([171, 165, 111])
            upper_red = np.array([179, 255, 211])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 165, 111])
            upper_red = np.array([10, 255, 211])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            mask=cv2.bitwise_or(mask1,mask2)
      

    if color=="b":

        if time.localtime().tm_hour<=18 and time.localtime().tm_hour>=6 :
            lower_b = np.array([106, 165, 28])
            upper_b = np.array([117, 250, 83])
        else:
            lower_b = np.array([108, 126, 124])
            upper_b = np.array([116, 175, 161])

        mask=cv2.inRange(hsv, lower_b, upper_b)


    result = cv2.bitwise_and(img, img, mask=mask)

    kernel = np.ones((27,27), np.uint8)
    erosion = cv2.erode(result, kernel, iterations = 1)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)


    _,contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    A_max=0
    c_max=None
    for c in contours:
        __, _, w1, h1 = cv2.boundingRect(c)
        if cv2.contourArea(c)>A_max and w1>50 and h1>70 and (1.0*h1/w1)<(3.0/2)*1.2 and (1.0*h1/w1)>(3.0/2)*0:
            A_max=cv2.contourArea(c)
            c_max=c
    
    x, y, w, h = cv2.boundingRect(c_max)
    # print(x,y,w,h)
   
    return x,y,w,h

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/tello_raw",Image,self.callback)
        self.bgimg = cv2.imread("bg.png")

    def callback(self,data):
        try:
            if not time.time()%5<5:
                return
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            x, y, w, h = findRect(cv_image,"r")
            # print(x,y,w,h)
            p=Point()
            p.x=x+w/2
            p.y=y+h/2
            p.z=w
            pub_r.publish(p)
            addbox=cv2.rectangle(cv_image, (x,y), (x+w,y+h), (0,0,255), 5) 

            x, y, w, h = findRect(cv_image,"g")
            # print(x,y,w,h)
            p=Point()
            p.x=x+w/2
            p.y=y+h/2
            p.z=w
            pub_g.publish(p)
            addbox=cv2.rectangle(cv_image, (x,y), (x+w,y+h), (0,255,0), 5) 

            x, y, w, h = findRect(cv_image,"b")
            # print(x,y,w,h)
            p=Point()
            p.x=x+w/2
            p.y=y+h/2
            p.z=w
            pub_b.publish(p)
            addbox=cv2.rectangle(cv_image, (x,y), (x+w,y+h), (255,0,0), 5) 
            addbox=cv2.circle(addbox, (480,360), 5, 255)

            imgAndState = np.hstack((addbox,self.bgimg))
            imgAndState = cv2.putText(imgAndState,str(power_last),(1130,200),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(box_x),(1130,280),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(box_y),(1130,320),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(box_z),(1130,360),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(x_d),(1130,435),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(y_d),(1130,475),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            imgAndState = cv2.putText(imgAndState,str(z_d),(1130,515),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 1, cv2.LINE_AA)
            cv2.imshow('addbox', imgAndState)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)


ic = image_converter()
rospy.init_node('image_converter', anonymous=True)
pub_r = rospy.Publisher('box_in_img_r', Point, queue_size=1)
pub_g = rospy.Publisher('box_in_img_g', Point, queue_size=1)
pub_b = rospy.Publisher('box_in_img_b', Point, queue_size=1)
ref_sub = rospy.Subscriber('ref', Twist, cb_ref)
box_sub = rospy.Subscriber('from_kf', Twist, cb_box)
power_sub = rospy.Subscriber('tello/status', TelloStatus, cb_power)
rospy.spin()