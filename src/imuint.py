#!/usr/bin/env python
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
import time
import rospy
import tf
rospy.init_node('int_imu', anonymous=True)
px = 1.5
py = 0
pz = 0


imu_pub = rospy.Publisher("from_IMU", PoseStamped, queue_size=1)
imu_pub2 = rospy.Publisher("from_IMU2", Twist, queue_size=1)
last_time=None
def callback(data):
    global px, py, pz, last_time, imu_pub
    t=time.time()
    if last_time is None:
        last_time = t 
    dt = t-last_time
    last_time = t
    px = -data.twist.twist.angular.x*dt+px
    py = data.twist.twist.angular.y*dt+py
    pz = -data.twist.twist.angular.z*dt+pz
    #print(px,dt,data.twist.twist.angular.x)
    q0 = data.pose.pose.orientation.w
    q1 = data.pose.pose.orientation.x
    q2 = data.pose.pose.orientation.y
    q3 = data.pose.pose.orientation.z
    pubmsg = PoseStamped()
    pubmsg.pose.position.x = px
    pubmsg.pose.position.y = py
    pubmsg.pose.position.z = pz
    pubmsg.pose.orientation.w = q0
    pubmsg.pose.orientation.x = q1
    pubmsg.pose.orientation.y = q2
    pubmsg.pose.orientation.z = q3
    pubmsg.header = data.header
    imu_pub.publish(pubmsg)
    pubmsg=Twist()
    pubmsg.linear.x=px
    pubmsg.linear.y=py
    pubmsg.linear.z=pz
    imu_pub2.publish(pubmsg)

imu_sub = rospy.Subscriber("tello/odom", Odometry, callback)

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
