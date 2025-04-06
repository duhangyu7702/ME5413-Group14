#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry
import math

def odom_callback(msg):
    # 从 odom 消息中提取线速度分量
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    vz = msg.twist.twist.linear.z
    # 计算线速度的模值
    linear_speed = math.sqrt(vx**2 + vy**2 + vz**2)
    rospy.loginfo("当前线速度: {:.2f} m/s".format(linear_speed))

def main():
    rospy.init_node('speed_printer', anonymous=True)
    # 订阅 odom 消息（如果没有 odom，可以改为订阅 /cmd_vel）
    rospy.Subscriber('/cmd_vel', Odometry, odom_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
