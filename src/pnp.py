#!/usr/bin/env python
import cv2
import os
import numpy as np

def mypnp(p):
    # (x1,y1)  (x3,y3)
    # (x4,y4)  (x2,y2)
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]=p
    image_points = np.array([
                            (x1,y1),
                            (x3,y3),
                            (x4,y4),
                            (x2,y2)
                        ], dtype="double")
    model_points = np.array([
                            (0.0,-0.1,0.15),
                            (0.0,0.1,0.15),
                            (0.0,-0.1,-0.15),
                            (0.0,0.1,-0.15)
                            ], dtype="double")

    cameraMatrix=np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
    distCoeffs=np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    a=rotation_vector
    b=translation_vector
    rr, _ = cv2.Rodrigues(np.array([a[0][0], a[1][0], a[2][0]]))
    tt = np.array([b[0][0], b[1][0], b[2][0]])
    cam_r = rr.transpose()
    cam_t = -cam_r.dot(tt)
    cam_x = cam_t[0]
    cam_y = cam_t[1]
    cam_z = cam_t[2]
    return cam_t,cam_r,tt,rr

def mypnpTotal(pr,pg,pb):
    # (x1,y1)  (x3,y3)
    # (x4,y4)  (x2,y2)
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]=pr
    [[x11,y11],[x12,y12],[x13,y13],[x14,y14]]=pg
    [[x21,y21],[x22,y22],[x23,y23],[x24,y24]]=pb
    image_points = np.array([
                            (x1,y1),
                            (x3,y3),
                            (x4,y4),
                            (x2,y2),
                            (x11,y11),
                            (x13,y13),
                            (x14,y14),
                            (x12,y12),
                            (x21,y21),
                            (x23,y23),
                            (x24,y24),
                            (x22,y22)
                        ], dtype="double")
    model_points = np.array([
                            (0.0,-0.1,0.15),
                            (0.0,0.1,0.15),
                            (0.0,-0.1,-0.15),
                            (0.0,0.1,-0.15),
                            (0.0,-0.4,0.15),
                            (0.0,-0.2,0.15),
                            (0.0,-0.4,-0.15),
                            (0.0,-0.2,-0.15),
                            (0.0,-0.4,0.55),
                            (0.0,-0.2,0.55),
                            (0.0,-0.4,0.25),
                            (0.0,-0.2,0.25)
                            ], dtype="float32")

    cameraMatrix=np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
    distCoeffs=np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])
    distCoeffs=np.array([0.000000, 0.000000, 0.000000,0.000000, 0.000000])
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    a=rotation_vector
    b=translation_vector
    rr, _ = cv2.Rodrigues(np.array([a[0][0], a[1][0], a[2][0]]))
    tt = np.array([b[0][0], b[1][0], b[2][0]])
    cam_r = rr.transpose()
    cam_t = -cam_r.dot(tt)
    cam_x = cam_t[0]
    cam_y = cam_t[1]
    cam_z = cam_t[2]
    rr, _ = cv2.Rodrigues(rotation_vector)
    return cam_t,cam_r,tt,rr

def dis3D(x1,y1,z1,x2,y2,z2):
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))

def ang3D(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    v1=[x1-x2,y1-y2,z1-z2]
    v2=[x3-x2,y3-y2,z3-z2]
    a=v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
    b=np.sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2])
    b=b*np.sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2])
    return np.arccos(a/b)/np.pi*180

if __name__ == '__main__':
    
    p=[[485,304],[548,438],[550,293],[486,462]]
    print(mypnp(p))
    