#!/usr/bin/env python
import numpy as np
import time
import matplotlib.pyplot as plt

TASKPOINT_NUM=2
ALPHA_H=0.6119
ALPHA_V=0.4845
Z_T=1.5
Z_B=4.5
THETA_A=np.pi/3
RHO=0.01


taskPoint=[[0.0,0.0,0.0,0.0] for i in range(TASKPOINT_NUM)]

def cameraFrame2WorldFrame(cpk,ci):
    theta=ci[3]
    pk=[0.0,0.0,0.0,0.0]
    pk[0]=cpk[0]*np.cos(theta)-cpk[2]*np.sin(theta)+ci[0]
    pk[1]=cpk[0]*np.sin(theta)+cpk[2]*np.cos(theta)+ci[1]
    pk[2]=(-1.0)*cpk[1]+ci[2]
    pk[3]=cpk[3]+theta
    return pk

def worldFrame2CameraFrame(pk,ci):
    theta=ci[3]
    cpk=[0.0,0.0,0.0,0.0]
    t=[0.0,0.0,0.0]
    t[0]=pk[0]-ci[0]
    t[1]=pk[1]-ci[1]
    t[2]=pk[2]-ci[2]
    cpk[0]=t[0]*np.cos(theta)+t[1]*np.sin(theta)
    cpk[1]=(-1.0)*t[2]
    cpk[2]=(-1.0)*t[0]*np.sin(theta)+t[1]*np.cos(theta)
    cpk[3]=pk[3]-theta
    return cpk

def ciSpace2tiSpace(cpk):
    tpk=[0.0,0.0,0.0,0.0]
    tpk[0]=cpk[0]/(cpk[2]*np.tan(ALPHA_H))
    tpk[1]=cpk[1]/(cpk[2]*np.tan(ALPHA_V))
    tpk[2]=(2.0*cpk[2]-Z_T-Z_B)/(Z_B-Z_T)
    tpk[3]=np.tan(abs(cpk[3])*0.5)/np.tan(THETA_A*0.5)
    return tpk

def d_v(pk,ci):
    cpk=worldFrame2CameraFrame(pk,ci)
    tpk=ciSpace2tiSpace(cpk)
    # print("max",tpk[0],tpk[1],tpk[2],tpk[3])
    return max(abs(tpk[0]),abs(tpk[1]),abs(tpk[2]),abs(tpk[3]))

def C_s(pk,ci):
    return np.exp((-1.0)*RHO*d_v(pk,ci))

def whatPartition(pk,ci):
    cpk=worldFrame2CameraFrame(pk,ci)
    tpk=ciSpace2tiSpace(cpk)
    if d_v(pk,ci)==tpk[0]:
        return 1
    elif d_v(pk,ci)==(-1.0)*tpk[0]:
        return 2
    elif d_v(pk,ci)==tpk[1]:
        return 3
    elif d_v(pk,ci)==(-1.0)*tpk[1]:
        return 4
    elif d_v(pk,ci)==tpk[2]:
        return 5
    elif d_v(pk,ci)==(-1.0)*tpk[2]:
        return 6
    elif d_v(pk,ci)==tpk[3]:
        return 7

def Partial_C_s_Partial_ci(pk,ci,partition):
    _Partial_C_s_Partial_ci=[0.0,0.0,0.0,0.0]
    partial_C_s_partial_dv=(-1.0)*RHO*np.exp((-1.0)*RHO*d_v(pk,ci))
    cpk=worldFrame2CameraFrame(pk,ci)
    if partition==7:
        if cpk[3]<0:
            Partial_dv_Partial_ci_3=(1.0/(2.0*np.tan(THETA_A/2)))*(1.0/(np.cos(cpk[3]/2)*np.cos(cpk[3]/2)))
        else:
            Partial_dv_Partial_ci_3=(1.0/(2.0*np.tan(THETA_A/2)))*(1.0/(np.cos(cpk[3]/2)*np.cos(cpk[3]/2)))*(-1)

        _Partial_C_s_Partial_ci=[0,0,0,partial_C_s_partial_dv*Partial_dv_Partial_ci_3]
    else:
        Partial_cpk_Partial_theta=[0.0,0.0,0.0]
        Partial_cpk_Partial_theta[0]=np.sin(pk[3])*(-1.0)*(pk[0]-ci[0])-np.cos(pk[3])*(pk[2]-ci[2])
        Partial_cpk_Partial_theta[1]=np.cos(pk[3])*(pk[0]-ci[0])-np.sin(pk[3])*(pk[2]-ci[2])
        # Partial_cpk_Partial_ci = [-R^T , Partial_cpk_Partial_theta]_{3*4}

        Partial_dv_Partial_cpk=[0.0,0.0,0.0]
        if partition==1:
            Partial_dv_Partial_cpk[0]=1.0/(cpk[2]*np.tan(ALPHA_H))
            Partial_dv_Partial_cpk[1]=0.0
            Partial_dv_Partial_cpk[2]=(-1.0*cpk[0])/(cpk[2]*cpk[2]*np.tan(ALPHA_H))
        elif partition==2:
            Partial_dv_Partial_cpk[0]=(-1.0)/(cpk[2]*np.tan(ALPHA_H))
            Partial_dv_Partial_cpk[1]=0.0
            Partial_dv_Partial_cpk[2]=(1.0*cpk[0])/(cpk[2]*cpk[2]*np.tan(ALPHA_H))
        elif partition==3:
            Partial_dv_Partial_cpk[0]=0.0
            Partial_dv_Partial_cpk[1]=(1.0)/(cpk[2]*np.tan(ALPHA_V))
            Partial_dv_Partial_cpk[2]=(-1.0*cpk[1])/(cpk[2]*cpk[2]*np.tan(ALPHA_V))
        elif partition==4:
            Partial_dv_Partial_cpk[0]=0.0
            Partial_dv_Partial_cpk[1]=(-1.0)/(cpk[2]*np.tan(ALPHA_V))
            Partial_dv_Partial_cpk[2]=(1.0*cpk[1])/(cpk[2]*cpk[2]*np.tan(ALPHA_V))
        elif partition==5:
            Partial_dv_Partial_cpk[0]=0.0
            Partial_dv_Partial_cpk[1]=0.0
            Partial_dv_Partial_cpk[2]=(2.0)/(Z_B-Z_T)
        elif partition==6:
            Partial_dv_Partial_cpk[0]=0.0
            Partial_dv_Partial_cpk[1]=0.0
            Partial_dv_Partial_cpk[2]=(-2.0)/(Z_B-Z_T)
        
        Partial_dv_Partial_ci=[0.0,0.0,0.0,0.0]
        Partial_dv_Partial_ci[0]=Partial_dv_Partial_cpk[0]*(-np.cos(pk[3]))+Partial_dv_Partial_cpk[2]*(np.sin(pk[3]))
        Partial_dv_Partial_ci[1]=Partial_dv_Partial_cpk[0]*(-np.sin(pk[3]))-Partial_dv_Partial_cpk[2]*(np.cos(pk[3]))
        Partial_dv_Partial_ci[2]=Partial_dv_Partial_cpk[1]
        Partial_dv_Partial_ci[3]=Partial_dv_Partial_cpk[0]*Partial_cpk_Partial_theta[0]+Partial_dv_Partial_cpk[1]*Partial_cpk_Partial_theta[1]

        _Partial_C_s_Partial_ci[0]=partial_C_s_partial_dv*Partial_dv_Partial_ci[0]
        _Partial_C_s_Partial_ci[1]=partial_C_s_partial_dv*Partial_dv_Partial_ci[1]
        _Partial_C_s_Partial_ci[2]=partial_C_s_partial_dv*Partial_dv_Partial_ci[2]
        _Partial_C_s_Partial_ci[3]=partial_C_s_partial_dv*Partial_dv_Partial_ci[3]

    return _Partial_C_s_Partial_ci


def gradient(pk,ci,it_num,it_length):
    partition=whatPartition(pk,ci)
    all_delta=[0.0,0.0,0.0,0.0]
    for i in range(it_num):
        delta=Partial_C_s_Partial_ci(pk,ci,partition)
        ci[0]=ci[0]+it_length*delta[0]
        ci[1]=ci[1]+it_length*delta[1]
        ci[2]=ci[2]+it_length*delta[2]
        ci[3]=ci[3]+it_length*delta[3]
        all_delta[0]=delta[0]
        all_delta[1]=delta[1]
        all_delta[2]=delta[2]
        all_delta[3]=delta[3]
    return ci,all_delta

def mut_point(ci,taskPoint):
    all_delta_ci=[0.0,0.0,0.0,0.0]
    for i in range(len(taskPoint)):
        partition=whatPartition(taskPoint[i],ci)
        # print(str(i),partition)
        delta=Partial_C_s_Partial_ci(taskPoint[i],ci,partition)
        all_delta_ci[0]=delta[0]+all_delta_ci[0]
        all_delta_ci[1]=delta[1]+all_delta_ci[1]
        all_delta_ci[2]=delta[2]+all_delta_ci[2]
        all_delta_ci[3]=delta[3]+all_delta_ci[3]
        # print(str(i),delta)
    return all_delta_ci


def v_y(x,y,th,l=-0.1):
    xx=x+np.cos(th+np.pi*0.5)*l
    yy=y+np.sin(th+np.pi*0.5)*l
    return [x,xx],[y,yy]

if __name__ == '__main__':
    taskPoint[0]=[0.0,0.0,0.0,np.pi/2]
    taskPoint[1]=[0.0,1.0,0.0,np.pi/2]
    ci=[1.0,1.0,0.0,np.pi/2]
    it_length=0.01

    for i in range(100):
        delta_ci=mut_point(ci,taskPoint)
        # print(delta_ci)
        ci[0]=ci[0]+it_length*delta_ci[0]
        ci[1]=ci[1]+it_length*delta_ci[1]
        ci[2]=ci[2]+it_length*delta_ci[2]
        ci[3]=ci[3]+it_length*0.1*delta_ci[3]
        length=delta_ci[0]*delta_ci[0]+delta_ci[1]*delta_ci[1]+delta_ci[2]*delta_ci[2]+delta_ci[3]*delta_ci[3]*0.01
        # print(ci,length)
    
    cpk=worldFrame2CameraFrame(taskPoint[0],ci)
    tpk=ciSpace2tiSpace(cpk)
    print(tpk)
    cpk=worldFrame2CameraFrame(taskPoint[1],ci)
    tpk=ciSpace2tiSpace(cpk)
    print(tpk)


    from geometry_msgs.msg import Twist

    target1=Twist()
    target1.linear.x=taskPoint[0][0]
    target1.linear.y=taskPoint[0][1]
    target1.linear.z=taskPoint[0][2]
    target1.angular.z=taskPoint[0][3]
    target2=Twist()
    target2.linear.x=taskPoint[1][0]
    target2.linear.y=taskPoint[1][1]
    target2.linear.z=taskPoint[1][2]
    target2.angular.z=taskPoint[1][3]
    
    
    res=Twist()
    res.linear.x=ci[0]
    res.linear.y=ci[1]
    res.linear.z=ci[2]
    res.angular.z=ci[3]

    import matplotlib.pyplot as plt
    plt.scatter(target1.linear.x,target1.linear.y)
    [x,xx],[y,yy]=v_y(target1.linear.x,target1.linear.y,target1.angular.z)
    plt.plot([x,xx],[y,yy])
    plt.scatter(target2.linear.x,target2.linear.y)
    [x,xx],[y,yy]=v_y(target2.linear.x,target2.linear.y,target2.angular.z)
    plt.plot([x,xx],[y,yy])
    # plt.scatter((target2.linear.x+target1.linear.x)*0.5,(target2.linear.y+target1.linear.y)*0.5)
    # plt.plot([target1.linear.x,target2.linear.x],[target1.linear.y,target2.linear.y])
    plt.scatter(res.linear.x,res.linear.y)
    [x,xx],[y,yy]=v_y(res.linear.x,res.linear.y,res.angular.z,0.3)
    plt.plot([x,xx],[y,yy])    
    # plt.axis([-3,3,-3,3])
    plt.grid(True)

    plt.show()

    x_list=np.linspace(3,3,1)
    y_list=np.linspace(-1,2,301)
    the_MAX=0
    for i in range(len(x_list)):
        for j in range(len(y_list)):
            ci=[x_list[i],y_list[j],0,np.pi/2]
            pk1=taskPoint[0]
            pk2=taskPoint[1]
            if C_s(pk1,ci)+C_s(pk2,ci)>the_MAX:
                the_MAX=C_s(pk1,ci)+C_s(pk2,ci)
            print(x_list[i],y_list[j],C_s(pk1,ci)+C_s(pk2,ci))