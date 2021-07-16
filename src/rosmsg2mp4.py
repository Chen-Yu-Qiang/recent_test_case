#!/usr/bin/env python
import os
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty
import datetime
import os
import random
def secnsec2s(sec,nsec):
    return 1.0*sec+nsec*(10**-9)

t0=0
t_tf=0
def cb_imu(data):
    global t0
    if t_tf==0:
        t0=secnsec2s(data.header.stamp.secs,data.header.stamp.nsecs)

def cb_tf(data):
    global t0,t_tf
    t_tf=t0


r=random.randint(1,100)


class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/tello_raw",Image,self.callback)
        length = 30
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.output_movie = cv2.VideoWriter('output'+str(r)+'.mp4', fourcc, length, (960, 720))
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            t=secnsec2s(data.header.stamp.secs,data.header.stamp.nsecs)
            if not t_tf==0:
                cv_image = cv2.putText(cv_image,"{:<6.2f}".format(t-t_tf),(0,75),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
            t2=datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d')
            cv_image = cv2.putText(cv_image,t2,(0,25),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
            t2=datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')
            cv_image = cv2.putText(cv_image,t2+"."+str(int(data.header.stamp.nsecs/(10**7))),(0,50),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
            self.output_movie.write(cv_image)
            cv2.imshow('addbox', cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
    def mydel(self):
        self.output_movie.release()
        cv2.destroyAllWindows()
        if t0==0:
            print("not start")
            os.remove('output'+str(r)+'.mp4')
        else:
            print("Save as "+datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')+".mp4")
            os.rename('output'+str(r)+'.mp4',datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')+".mp4")
        print("886")


ic = image_converter()
rospy.init_node('image_record', anonymous=True)
tf_sub = rospy.Subscriber('tello/takeoff', Empty, cb_tf)
imu_sub = rospy.Subscriber('tello/imu', Imu, cb_imu)
rospy.spin()
ic.mydel()