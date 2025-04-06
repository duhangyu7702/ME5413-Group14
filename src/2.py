#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：订阅ROS发布的两路图像数据，并将其分别保存为视频文件到本地。
依赖：ROS (rospy, sensor_msgs)、cv_bridge、opencv-python
使用方法：将此文件放到ROS工作空间中，修改订阅话题名称和参数后作为节点运行。
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class DualImageToVideo:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('dual_image_to_video_node', anonymous=True)

        # cv_bridge用于在ROS图像消息和OpenCV图像之间转换
        self.bridge = CvBridge()

        # 获取参数：帧率、视频保存路径以及订阅的话题名称
        self.fps = rospy.get_param("~fps", 30)

        self.video_filename1 = rospy.get_param("~video_filename1", "left.mp4")
        self.video_filename2 = rospy.get_param("~video_filename2", "left2.mp4")

        self.image_topic1 = rospy.get_param("~image_topic1", "/left_camera/image_raw")
        self.image_topic2 = rospy.get_param("~image_topic2", "/left_camera_2/image_raw")

        # 初始化VideoWriter及相关变量
        self.video_writer1 = None
        self.video_writer2 = None
        self.frame_size1 = None
        self.frame_size2 = None

        # 订阅两个图像话题
        self.subscriber1 = rospy.Subscriber(self.image_topic1, Image, self.callback1)
        self.subscriber2 = rospy.Subscriber(self.image_topic2, Image, self.callback2)

        rospy.loginfo("已订阅图像话题: %s", self.image_topic1)
        rospy.loginfo("视频将保存到: %s", self.video_filename1)
        rospy.loginfo("已订阅图像话题: %s", self.image_topic2)
        rospy.loginfo("视频将保存到: %s", self.video_filename2)

    def callback1(self, data):
        try:
            # 将ROS图像消息转换为OpenCV格式（BGR8）
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge转换失败（话题1）: %s", e)
            return

        # 如果尚未初始化VideoWriter，则根据图像尺寸进行初始化
        if self.video_writer1 is None:
            self.frame_size1 = (cv_image.shape[1], cv_image.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer1 = cv2.VideoWriter(self.video_filename1, fourcc, self.fps, self.frame_size1)
            rospy.loginfo("VideoWriter1已初始化，尺寸: %s, fps: %d", self.frame_size1, self.fps)

        # 将接收到的图像帧写入视频文件
        self.video_writer1.write(cv_image)

    def callback2(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge转换失败（话题2）: %s", e)
            return

        if self.video_writer2 is None:
            self.frame_size2 = (cv_image.shape[1], cv_image.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer2 = cv2.VideoWriter(self.video_filename2, fourcc, self.fps, self.frame_size2)
            rospy.loginfo("VideoWriter2已初始化，尺寸: %s, fps: %d", self.frame_size2, self.fps)

        self.video_writer2.write(cv_image)

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("节点关闭中...")
        finally:
            if self.video_writer1:
                self.video_writer1.release()
            if self.video_writer2:
                self.video_writer2.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    node = DualImageToVideo()
    node.run()
