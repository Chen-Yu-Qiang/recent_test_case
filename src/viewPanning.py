#!/usr/bin/env python
import numpy as np
import time
import matplotlib.pyplot as plt

TASKPOINT_NUM=2
ALPHA_H=0.6119
ALPHA_V=0.4845
Z_T=1.5
Z_B=4.5
THETA_A=np.pi/4
RHO=1


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
    if d_v(pk,ci)>100:
        return 0
    else:
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
    partial_C_s_partial_dv=(-2.0)*d_v(pk,ci)*RHO*np.exp((-1.0)*RHO*d_v(pk,ci)*d_v(pk,ci))
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


def v_y(x,y,th,l=-0.5):
    xx=x+np.cos(th+np.pi*0.5)*l
    yy=y+np.sin(th+np.pi*0.5)*l
    return [x,xx],[y,yy]

if __name__ == '__main__':

    from geometry_msgs.msg import Twist
    import matplotlib.pyplot as plt

    TASKPOINT_NUM=2
    taskPoint=[[0.0,0.0,0.0,0.0] for i in range(TASKPOINT_NUM)]

    # Case board
    # taskPoint[0]=[0.0,0.0,0.0,np.pi/2]

    # Case 1
    taskPoint[0]=[0.0,0.65,0.0,np.pi/2]
    taskPoint[1]=[0.0,-0.8,0.0,np.pi*0.5]

    # Case 2
    # taskPoint[0]=[0.0,0.65,0.0,np.pi/2]
    # taskPoint[1]=[0.0,-0.8,0.0,np.pi*2.0/3]

    # Case 3
    # taskPoint[0]=[0.0,0.65,0.0,np.pi/2]
    # taskPoint[1]=[0.52,-0.8,0.0,np.pi*0.75]


    targetSet=[Twist() for i in range(TASKPOINT_NUM)]

    for i in range(len(taskPoint)):
        targetSet[i].linear.x=taskPoint[i][0]
        targetSet[i].linear.y=taskPoint[i][1]
        targetSet[i].linear.z=taskPoint[i][2]
        targetSet[i].angular.z=taskPoint[i][3]

    ci=[1.0,1.0,0.0,np.pi]
    it_length=1

    overall_per=[0 for i in range(1000)]
    t1_per=[0 for i in range(1000)]
    t2_per=[0 for i in range(1000)]
    trj_x=[0 for i in range(1000)]
    trj_y=[0 for i in range(1000)]
    tck1_1=[0 for i in range(1000)]
    tck1_2=[0 for i in range(1000)]
    tck1_3=[0 for i in range(1000)]
    tck1_4=[0 for i in range(1000)]
    tck2_1=[0 for i in range(1000)]
    tck2_2=[0 for i in range(1000)]
    tck2_3=[0 for i in range(1000)]
    tck2_4=[0 for i in range(1000)]

    board_color=["tab:orange","tab:green"]
    # for i in range(len(taskPoint)):
    #     plt.scatter(targetSet[i].linear.x,targetSet[i].linear.y,color=board_color[i])
    #     [x,xx],[y,yy]=v_y(targetSet[i].linear.x,targetSet[i].linear.y,targetSet[i].angular.z)
    #     plt.plot([x,xx],[y,yy],color=board_color[i])


    for i in range(1000):
        delta_ci=mut_point(ci,taskPoint)
        # print(delta_ci)
        ci[0]=ci[0]+it_length*delta_ci[0]
        ci[1]=ci[1]+it_length*delta_ci[1]
        ci[2]=ci[2]+it_length*delta_ci[2]
        ci[3]=ci[3]+it_length*0.1*delta_ci[3]
        it_length=it_length*0.999
        res=Twist()
        res.linear.x=ci[0]
        res.linear.y=ci[1]
        res.linear.z=ci[2]
        res.angular.z=ci[3]
        t1_per[i]=C_s(taskPoint[0],ci)
        # t2_per[i]=C_s(taskPoint[1],ci)
        overall_per[i]=t1_per[i]+t2_per[i]



        # for trajectory =====================================
        # trj_x[i]=ci[0]
        # trj_y[i]=ci[1]
        # plt.scatter(res.linear.x,res.linear.y,color="tab:blue")
        # [x,xx],[y,yy]=v_y(res.linear.x,res.linear.y,res.angular.z,0.3)
        # plt.plot([x,xx],[y,yy],color="tab:blue")    
        # ===========================================================        



        # for animation=====================================================================
        # plt.scatter(target1.linear.x,target1.linear.y)
        # [x,xx],[y,yy]=v_y(target1.linear.x,target1.linear.y,target1.angular.z)
        # plt.plot([x,xx],[y,yy])
        # plt.scatter(target2.linear.x,target2.linear.y)
        # [x,xx],[y,yy]=v_y(target2.linear.x,target2.linear.y,target2.angular.z)
        # plt.plot([x,xx],[y,yy])
        # plt.scatter(res.linear.x,res.linear.y)
        # [x,xx],[y,yy]=v_y(res.linear.x,res.linear.y,res.angular.z,0.3)
        # plt.plot([x,xx],[y,yy])    
        # plt.axis([-1,5,-1,2])
        # plt.grid(True)

        # plt.draw()
        # plt.pause(0.1)
        # plt.clf()
        # =================================================================================

        cpk=worldFrame2CameraFrame(taskPoint[0],ci)
        tpk=ciSpace2tiSpace(cpk)

        tck1_1[i]=tpk[0]
        tck1_2[i]=tpk[1]
        tck1_3[i]=tpk[2]
        tck1_4[i]=tpk[3]

        cpk=worldFrame2CameraFrame(taskPoint[1],ci)
        tpk=ciSpace2tiSpace(cpk)

        tck2_1[i]=tpk[0]
        tck2_2[i]=tpk[1]
        tck2_3[i]=tpk[2]
        tck2_4[i]=tpk[3]

        print("ci",ci,i)

    # plt.plot(trj_x,trj_y,color="b")
    # plt.axis([-1,5,-2,2])
    # plt.grid(True)
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # plt.show()

    plt.plot(tck1_1)
    plt.plot(tck1_2)
    plt.plot(tck1_3)
    plt.plot(tck1_4)
    plt.grid(True)
    plt.xlabel("Number of iterations")
    plt.ylabel(" (m)")
    plt.axis([0,1000,0,2])
    plt.legend(["tp_xk","tp_yk","tp_zk","tp_thk"])
    plt.show()


    plt.plot(tck2_1)
    plt.plot(tck2_2)
    plt.plot(tck2_3)
    plt.plot(tck2_4)
    plt.grid(True)
    plt.xlabel("Number of iterations")
    plt.ylabel(" (m)")
    plt.axis([0,1000,0,2])
    plt.legend(["tp_xk","tp_yk","tp_zk","tp_thk"])
    plt.show()




    # plt.figure()
    # plt.plot(t1_per)
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Performance")
    # plt.title("Target 51")
    # plt.grid(True)
    # plt.axis([0,1000,0,2])
    # plt.show()


    # plt.figure()
    # plt.plot(t2_per)
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Performance")
    # plt.title("Target 52")
    # plt.grid(True)
    # plt.axis([0,1000,0,2])
    # plt.show()
    
    # plt.figure()
    # plt.plot(overall_per)
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Performance")
    # plt.title("Overall")
    # plt.grid(True)
    # plt.axis([0,1000,0,2])
    # plt.show()







    # for contourf=======================================================
    # x_list=np.linspace(-1,5,401)
    # y_list=np.linspace(-2,2,401)

    # def f(x,y):
    #     C_s_sum=0
    #     _ci=[x,y,0,ci[3]]
    #     for i in range(len(taskPoint)):
    #         C_s_sum=C_s_sum+C_s(taskPoint[i],_ci)
    #         # if C_s(taskPoint[i],_ci)==0:
    #         #     return 0
    #     return C_s_sum


    # Z=[[0 for i in range(len(x_list))]for j in range(len(y_list))]
    # for i in range(len(x_list)):
    #      for j in range(len(y_list)):
    #          Z[j][i]=f(x_list[i],y_list[j])


    # plt.contourf(x_list, y_list, Z,100,cmap='jet')
    # plt.colorbar()    
    # plt.axis([-1,5,-2,2])
    # plt.scatter(res.linear.x,res.linear.y)
    # [x,xx],[y,yy]=v_y(res.linear.x,res.linear.y,res.angular.z,0.3)
    # plt.plot([x,xx],[y,yy])    

    # for i in range(len(taskPoint)):
    #     plt.scatter(targetSet[i].linear.x,targetSet[i].linear.y,color=board_color[i])
    #     [x,xx],[y,yy]=v_y(targetSet[i].linear.x,targetSet[i].linear.y,targetSet[i].angular.z)
    #     plt.plot([x,xx],[y,yy],color=board_color[i])

    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # plt.show()