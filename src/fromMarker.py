#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv2 import aruco
import numpy as np
import math


def quaternion_from_matrix(matrix):
    T1=np.append(matrix,np.array([[0,0,0]]),axis=0)
    T2=np.append(T1,np.array([[0],[0],[0],[1]]),axis=1)
    

    q = np.empty((4, ), dtype=np.float64)
    M = np.array(T2, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def qmulq(q1,q2):
    q3=[0,0,0,0]

    q3[0]=q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
    q3[1]=q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2]
    q3[2]=q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
    q3[3]=q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
    return q3


def ROSdata2CV(data):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        return cv_image
    except CvBridgeError as e:
        print(e)
        return

def cb_img(data):
    corners, ids, img = detect(ROSdata2CV(data))
    retval, rvec, tvec = est(corners, ids)
    if retval == 0:
        return 0, 0, 0
    a=rvec
    b=tvec
    rr, _ = cv2.Rodrigues(np.array([a[0][0], a[1][0], a[2][0]]))
    tt = np.array([b[0][0], b[1][0], b[2][0]])
    cam_r = rr.transpose()
    cam_t = -cam_r.dot(tt)
    cam_x = cam_t[0]
    cam_y = cam_t[1]
    cam_z = cam_t[2]
    cam_q=quaternion_from_matrix(cam_r)
    tq=[-0.5, 0.5, 0.5, 0.5]
    q3=qmulq(tq,cam_q)
    pubmsg=PoseStamped()
    pubmsg.pose.position.x=cam_x
    pubmsg.pose.position.y=cam_y
    pubmsg.pose.position.z=cam_z
    pubmsg.pose.orientation.w=q3[3]
    pubmsg.pose.orientation.x=q3[0]
    pubmsg.pose.orientation.y=q3[1]
    pubmsg.pose.orientation.z=q3[2]
    marker_pub.publish(pubmsg)


def detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img=cv2.GaussianBlur(img, (11, 11), 0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementMaxIterations=100
    params.cornerRefinementMinAccuracy=0.01
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=params)
    dr=img
    dr=aruco.drawDetectedMarkers(dr,corners,ids)

    cv2.imshow("a",dr)
    cv2.waitKey(1)
    return corners, ids, img


def est(corners, ids):
    ####for real tello by me
    # camera_matrix = np.array([[921.170702, 0.000000, 459.904354], [
    #    0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
    # dist_coeffs = np.array(
    #    [-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])


    ####for real tello by ROS
    camera_matrix = np.array([[937.878723, 0.000000, 489.753885], [
       0.000000, 939.156738, 363.172139], [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array(
       [-0.016272, 0.093492, 0, 0.002999, 0.000000])
    ####for sim
    # camera_matrix = np.array([[562, 0.000000, 480.5], [
    #     0.000000, 562, 360.5], [0.000000, 0.000000, 1.000000]])
    # dist_coeffs = np.array([0, 0, 0, 0, 0])
    

    thePoint = [[0, 0.11, 0.14],[0, 0.31, 0.14],[0, 0.31, -0.06],[0, 0.11, -0.06]]
    board_corners = [np.array(thePoint, dtype=np.float32)]
    board_ids = np.array([0], dtype=np.int32)
    board = aruco.Board_create(board_corners,
                               aruco.getPredefinedDictionary(
                               aruco.DICT_5X5_100),
                               board_ids)
    retval, rvec, tvec = aruco.estimatePoseBoard(
        corners, ids, board, camera_matrix, dist_coeffs, None, None)
    return retval, rvec, tvec


rospy.init_node('from_Marker', anonymous=True)
img_sub = rospy.Subscriber("tello_raw", Image, cb_img)
marker_pub=rospy.Publisher("from_Marker",PoseStamped,queue_size=1)

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")