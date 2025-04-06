#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped

def goal_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z
    rospy.loginfo("目标点坐标: x=%.2f, y=%.2f, z=%.2f", x, y, z)
    rospy.loginfo("目标点朝向: (%.2f, %.2f, %.2f, %.2f)",
                  msg.pose.orientation.x,
                  msg.pose.orientation.y,
                  msg.pose.orientation.z,
                  msg.pose.orientation.w)

def listener():
    rospy.init_node('goal_listener', anonymous=True)
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, goal_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
