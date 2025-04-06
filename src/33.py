#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import time


def image_callback(msg):
    bridge = CvBridge()
    try:
        # 将ROS Image消息转换为OpenCV图像（BGR格式）
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge转换错误: %s" % e)
        return

    # 获取保存路径（ROS参数~save_path, 如果未设置则使用当前目录）
    save_path = rospy.get_param("~save_path", os.getcwd())
    # 如果目录不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 生成保存文件的文件名，带上时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_path, "ros_image_" + timestamp + ".png")

    # 将图像保存到指定路径
    cv2.imwrite(filename, cv_image)
    rospy.loginfo("图像已保存至: %s" % filename)

    # 图像保存后退出节点，如果需要连续保存请去掉此行
    rospy.signal_shutdown("图像已保存，退出节点")


if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('ros_image_saver', anonymous=True)

    # 订阅图像话题（根据实际话题名称进行修改）
    rospy.Subscriber("/left_camera/image_raw", Image, image_callback)
    rospy.loginfo("等待接收图像消息...")

    rospy.spin()
