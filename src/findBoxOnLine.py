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

    ampm=14
    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    # print(hsv[360][480])
    if color=="g":
        if ampm==20:
            
            # 20 pm
            lower_g = np.array([40, 159, 139])
            upper_g = np.array([47, 223, 162])
        elif ampm==10:
            # 10am
            lower_g = np.array([42, 170, 51])
            upper_g = np.array([45, 223, 72])
        elif ampm== 14:
            # 14 pm
            lower_g = np.array([37, 180, 43])
            upper_g = np.array([43, 202, 56])
        elif ampm==25:
            lower_g = np.array([39, 62, 0])
            upper_g = np.array([70, 255, 255])

        mask=cv2.inRange(hsv, lower_g, upper_g)


    if color=="r":

        if ampm==20:
            # 20pm
            lower_red = np.array([176, 174, 119])
            upper_red = np.array([179, 255, 169])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 174, 119])
            upper_red = np.array([8, 255, 169])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            mask=cv2.bitwise_or(mask1,mask2)
        elif ampm==10:
        # 10 am
            lower_red = np.array([176, 215, 30])
            upper_red = np.array([179, 255, 55])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 215, 30])
            upper_red = np.array([2, 255, 55])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            mask=cv2.bitwise_or(mask1,mask2)
        elif ampm==14:
        # 14 pm
            lower_red = np.array([176, 234, 38])
            upper_red = np.array([179, 255, 52])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 234, 38])
            upper_red = np.array([2, 255, 52])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            mask=cv2.bitwise_or(mask1,mask2)   
        elif ampm==25:
            lower_red = np.array([176, 66, 0])
            upper_red = np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([0, 66, 0])
            upper_red = np.array([8, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            mask=cv2.bitwise_or(mask1,mask2)        


    if color=="b":

        if ampm==20:
            # 20pm
            lower_b = np.array([98, 140, 142])
            upper_b = np.array([102, 178, 165])        
        elif ampm==10:        
            # 10am
            lower_b = np.array([93, 160, 56])
            upper_b = np.array([104, 231, 79])        
        elif ampm==14:        
            # 14 pm
            lower_b = np.array([97, 174, 59])
            upper_b = np.array([104, 211, 74])      
        elif ampm==25:        
            # 10am
            lower_b = np.array([93, 56, 0])
            upper_b = np.array([104, 255, 255])
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