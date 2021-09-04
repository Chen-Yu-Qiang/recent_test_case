#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import time

class a_plan_res:
    def __init__(self,_name):
        self.str_name=_name
        self.sub_res=rospy.Subscriber('plan_wp_'+self.str_name, Twist,self.cb_res)
        self.sub_tpk=rospy.Subscriber('plan_tpk_'+self.str_name, Float32MultiArray,self.cb_tpk)
        self.res=Twist()
        self.tpk=None
        self.is_ok=1

        self.in_num=[51,52]
        self.in_num_obj=None

    def cb_res(self,data):
        # print(data)
        self.res=data


    def cb_tpk(self,data):
        self.tpk=data.data
        # print(data)
        if max(self.tpk)>=1:
            self.is_ok=0
        else:
            self.is_ok=1

    def which_board_timeout(self):
        for i in range(len(self.in_num_obj)):
            if self.in_num_obj[i].istimeout():
                return i
        return -1

    def get_output_msg(self):
        output_msg=Twist()

        if self.is_ok:
            if self.which_board_timeout()==-1:
                output_msg=self.res
                output_msg.angular.x=6
            else:
                output_msg=pr5152.in_num_obj[pr5152.which_board_timeout()].last_see_uav_pos
                output_msg.angular.x=0  

            return output_msg

class a_board:
    def __init__(self,_num):
        self.sub_res=rospy.Subscriber('target'+str(_num), Twist,self.cb_b)
        self.data=Twist()
        self.last_see_time=time.time()
        self.last_see_uav_pos=Twist()
        self.uav_pos=Twist()

    def cb_b(self,_data):
        self.data=_data
        self.last_see_uav_pos=self.uav_pos
        self.last_see_time=time.time()

    def istimeout(self):
        if time.time()-self.last_see_time>1:
            return 1
        else:
            return 0



board_set=[a_board(i) for i in [0,51,52]]

def cb_uav(data):
    global board_set
    for i in board_set:
        i.uav_pos=data

rospy.init_node('sw_plan_node', anonymous=True)


pr5152=a_plan_res("51_52")
pr5152.in_num_obj=[board_set[1],board_set[2]]
pr5152.in_num=[51,52]


pr52=a_plan_res("52")
pr52.in_num_obj=[board_set[2]]
pr52.in_num=[52]

rate = rospy.Rate(30)
p5152_pub=rospy.Publisher('ref_bef', Twist, queue_size=1)

#  51+52=110(2)=6(10)
#  52   =100(2)=4(10)
#  51   =010(2)=2(10)
is51or52_pub=rospy.Publisher('is51or52_pub', Twist, queue_size=1)
uav_sub=rospy.Subscriber('from_kf', Twist, cb_uav)
while not rospy.is_shutdown():
    if pr5152.is_ok:
        p5152_pub.publish(pr5152.get_output_msg())
            
    else:
        p5152_pub.publish(pr52.get_output_msg())
    
    rate.sleep()