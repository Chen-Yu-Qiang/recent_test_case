#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import filter_lib

rospy.init_node('ref_filter_node', anonymous=True)
ref_pub = rospy.Publisher('ref', Twist, queue_size=1)


f=filter_lib.meanFilter(21)
def ref_cb(data):
    global f,ref_pub
    ref_pub.publish(f.update(data))
    


takeoff_sub = rospy.Subscriber('ref_bef', Twist, ref_cb)



rospy.spin()