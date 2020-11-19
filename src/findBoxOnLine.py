#!/usr/bin/env python
import cv2
import os
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
def findRect(img,color):
    def nothing(data):
        pass

    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    # print(hsv[360][480])
    if color=="g":
        lower_g = np.array([30, 80, 80])
        upper_g = np.array([80, 255, 255])
        mask=cv2.inRange(hsv, lower_g, upper_g)


    if color=="r":
        lower_red = np.array([170, 80, 80])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = np.array([0, 80, 80])
        upper_red = np.array([10, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        mask=cv2.bitwise_or(mask1,mask2)

    if color=="b":
        lower_b = np.array([100, 40, 0])
        upper_b = np.array([120, 255, 255])
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
        if cv2.contourArea(c)>A_max:
            A_max=cv2.contourArea(c)
            c_max=c
    
    x, y, w, h = cv2.boundingRect(c_max)
    # print(x,y,w,h)
   
    return x,y,w,h

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/tello_raw",Image,self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            x, y, w, h = findRect(cv_image,"r")
            # print(x,y,w,h)
            p=Point()
            p.x=x+w/2
            p.y=y+h/2
            p.z=w
            pub_r.publish(p)
            addbox=cv2.rectangle(cv_image, (x,y), (x+w,y+h), (255,0,0), 5) 

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
            addbox=cv2.rectangle(cv_image, (x,y), (x+w,y+h), (0,0,255), 5) 
            addbox=cv2.circle(addbox, (480,360), 5, 255)


            cv2.imshow('addbox', addbox)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)


ic = image_converter()
rospy.init_node('image_converter', anonymous=True)
pub_r = rospy.Publisher('box_in_img_r', Point, queue_size=10)
pub_g = rospy.Publisher('box_in_img_g', Point, queue_size=10)
pub_b = rospy.Publisher('box_in_img_b', Point, queue_size=10)
rospy.spin()