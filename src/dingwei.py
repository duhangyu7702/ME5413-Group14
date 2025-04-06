#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry

def odom_callback(msg):
    # 提取位置信息
    pos = msg.pose.pose.position
    # 提取朝向信息
    ori = msg.pose.pose.orientation
    rospy.loginfo("Jackal 当前位姿: x=%.2f, y=%.2f, z=%.2f", pos.x, pos.y, pos.z)
    rospy.loginfo("Jackal 当前朝向: (x=%.2f, y=%.2f, z=%.2f, w=%.2f)",
                  ori.x, ori.y, ori.z, ori.w)

def listener():
    rospy.init_node('jackal_pose_listener', anonymous=True)
    # 订阅 /jackal_velocity_controller/odom 话题
    rospy.Subscriber('/jackal_velocity_controller/odom', Odometry, odom_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
