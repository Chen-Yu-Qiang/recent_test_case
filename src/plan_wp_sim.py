#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist
import time

rospy.init_node('sim_target', anonymous=True)
target51_pub = rospy.Publisher('target51', Twist, queue_size=1)
target52_pub = rospy.Publisher('target52', Twist, queue_size=1)


rate = rospy.Rate(30)
t51=Twist()
t51.linear.x=0
t51.linear.y=0.65
t51.linear.z=0.9
t51.angular.z=1.571

t52=Twist()
t52.linear.x=0.5
t52.linear.y=-0.65
t52.linear.z=0.9
t52.angular.z=1.571
i=0
while  not rospy.is_shutdown():

    target51_pub.publish(t51)
    t52.angular.z=2.356-i
    i=i+0.001
    print(i)
    # target52_pub.publish(t52)

    # t52.linear.x=t52.linear.x+0.01
    # i=i+0.01
    # print(i)
    target52_pub.publish(t52)

    rate.sleep()

