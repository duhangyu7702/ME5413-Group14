#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class GoalPublisher(object):
    def __init__(self):
        rospy.init_node('goal_publisher', anonymous=True)

        # 发布目标点 -> move_base_simple/goal
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # 订阅 AMCL 发布的位姿 (PoseWithCovarianceStamped)
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)

        # 控制机器人旋转 -> /cmd_vel（本例主要通过发布目标点来控制旋转）
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # 构造初始目标点消息（坐标和角度）
        self.goal_pose = PoseStamped()
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.pose.position.x = 20.56
        self.goal_pose.pose.position.y = -21.99
        # 初始角度对应的四元数（这里只设置 z 和 w，其它分量为 0）
        self.goal_pose.pose.orientation.z = 0.30
        self.goal_pose.pose.orientation.w = 0.95

        # 标记：当 /amcl_pose 停止发布 3 秒后进入拍照流程
        self.reached = False

        # 记录最近一次接收 /amcl_pose 的时间
        self.last_amcl_time = None
        self.stop_duration_required = 3.0  # 3 秒无消息则认为已停止

        # 用于将 ROS 图像转换为 OpenCV 图像
        self.bridge = CvBridge()

        # 从初始四元数计算当前 yaw 值
        qz = self.goal_pose.pose.orientation.z
        qw = self.goal_pose.pose.orientation.w
        self.goal_yaw = self.quaternion_to_yaw(0.0, 0.0, qz, qw)

        # 延时 3 秒，确保系统初始化并开始接收 /amcl_pose 消息
        rospy.loginfo("等待系统初始化 3 秒...")
        rospy.sleep(3.0)
        self.last_amcl_time = rospy.Time.now().to_sec()
        rospy.loginfo("系统初始化完成，开始正常运行。")

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """
        将四元数转换为欧拉角，然后返回 yaw。
        """
        (roll, pitch, yaw) = euler_from_quaternion([qx, qy, qz, qw])
        return yaw

    def quaternion_from_yaw(self, yaw, roll=0.0, pitch=0.0):
        """
        将 yaw 转换为四元数（假设 roll 和 pitch 为 0）。
        """
        return quaternion_from_euler(roll, pitch, yaw)

    def pose_callback(self, pose_msg):
        """
        回调函数：更新最近一次接收 /amcl_pose 的时间。
        """
        self.last_amcl_time = rospy.Time.now().to_sec()

    def capture_images(self):
        """
        分别从两个摄像头话题获取图像，并转换为 OpenCV 格式图像。
        """
        left_img_msg = rospy.wait_for_message('/left_camera/image_raw', Image)
        right_img_msg = rospy.wait_for_message('/left_camera_2/image_raw', Image)
        left_cv = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')
        right_cv = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='bgr8')
        return left_cv, right_cv

    def stitch_six_images(self, images):
        """
        利用 OpenCV 拼接器对六张图像进行全景拼接。
        拼接顺序：[left0, right0, left1, right1, left2, right2]
        """
        stitcher = cv2.Stitcher_create()
        status, stitched = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            rospy.loginfo("全景图像拼接成功")
            return stitched
        else:
            rospy.logerr("全景图像拼接失败，错误码为: %d", status)
            return None

    def run(self):
        """
        主要流程：
         1. 持续发布目标点；
         2. 当检测到 /amcl_pose 停止发布 3 秒后，进入拍照流程：
            a. 拍摄第一组图像，并保存到本地；
            b. 拍摄结束后保存到本地，然后等待 1 秒，
               然后通过话题发布新的坐标和角度，使机器人旋转 30°，
               发布后等待 5 秒；
            c. 拍摄第二组图像，并保存到本地；
            d. 拍摄结束后保存到本地，然后等待 1 秒，
               再次通过话题发布新的坐标和角度，使机器人再旋转 30°，
               发布后等待 5 秒；
            e. 拍摄第三组图像，并保存到本地；
         3. 拼接这六幅图像，并保存到本地。
        """
        rospy.loginfo("开始持续发布目标点...")
        rate = rospy.Rate(10)  # 发布频率 10Hz
        while not rospy.is_shutdown() and not self.reached:
            self.goal_pose.header.stamp = rospy.Time.now()
            self.goal_pub.publish(self.goal_pose)

            current_time = rospy.Time.now().to_sec()
            if self.last_amcl_time is not None:
                if (current_time - self.last_amcl_time) >= self.stop_duration_required:
                    rospy.loginfo("/amcl_pose 已停止发布 3 秒，进入拍照流程...")
                    self.reached = True

            rate.sleep()

        if not self.reached:
            rospy.logwarn("节点被关闭或其它原因终止，未检测到 /amcl_pose 停止。")
            return

        rospy.loginfo("开始拍摄全景所需图像...")

        # 1. 拍摄第一组图像
        rospy.loginfo("拍摄第一组图像...")
        left0, right0 = self.capture_images()
        cv2.imwrite("left0.jpg", left0)
        cv2.imwrite("right0.jpg", right0)
        rospy.loginfo("第一组图像已保存为 left0.jpg 和 right0.jpg")

        # 等待 1 秒
        rospy.loginfo("第一组图像拍摄完成，等待 1 秒...")
        rospy.sleep(1.0)

        # 2. 发布新的坐标和角度，使机器人旋转 30°（第一次旋转）
        rospy.loginfo("发布新的坐标和角度，使机器人旋转 30°...")
        new_yaw = self.goal_yaw + math.radians(30)
        self.goal_yaw = new_yaw  # 更新内部存储的 yaw 值
        new_quat = self.quaternion_from_yaw(new_yaw)
        self.goal_pose.pose.orientation.x = new_quat[0]
        self.goal_pose.pose.orientation.y = new_quat[1]
        self.goal_pose.pose.orientation.z = new_quat[2]
        self.goal_pose.pose.orientation.w = new_quat[3]
        self.goal_pose.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.goal_pose)

        # 发布完成后等待 5 秒
        rospy.loginfo("发布完成后等待 5 秒...")
        rospy.sleep(5.0)

        # 3. 拍摄第二组图像
        rospy.loginfo("拍摄第二组图像...")
        left1, right1 = self.capture_images()
        cv2.imwrite("left1.jpg", left1)
        cv2.imwrite("right1.jpg", right1)
        rospy.loginfo("第二组图像已保存为 left1.jpg 和 right1.jpg")

        # 等待 1 秒
        rospy.loginfo("第二组图像拍摄完成，等待 1 秒...")
        rospy.sleep(1.0)

        # 4. 再次发布新的坐标和角度，使机器人再旋转 30°（第二次旋转）
        rospy.loginfo("再次发布新的坐标和角度，使机器人旋转 30°...")
        new_yaw = self.goal_yaw + math.radians(30)
        self.goal_yaw = new_yaw
        new_quat = self.quaternion_from_yaw(new_yaw)
        self.goal_pose.pose.orientation.x = new_quat[0]
        self.goal_pose.pose.orientation.y = new_quat[1]
        self.goal_pose.pose.orientation.z = new_quat[2]
        self.goal_pose.pose.orientation.w = new_quat[3]
        self.goal_pose.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.goal_pose)

        # 发布完成后等待 5 秒
        rospy.loginfo("发布完成后等待 5 秒...")
        rospy.sleep(5.0)

        # 5. 拍摄第三组图像
        rospy.loginfo("拍摄第三组图像...")
        left2, right2 = self.capture_images()
        cv2.imwrite("left2.jpg", left2)
        cv2.imwrite("right2.jpg", right2)
        rospy.loginfo("第三组图像已保存为 left2.jpg 和 right2.jpg")

        # 6. 拼接六张图像
        rospy.loginfo("开始拼接全景图像...")
        images = [left0, right0, left1, right1, left2, right2]
        stitched = self.stitch_six_images(images)

        if stitched is not None:
            cv2.imwrite("p.jpg", stitched)
            rospy.loginfo("全景图像已保存为 p.jpg")
            cv2.imshow('Panorama', stitched)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            rospy.logerr("全景图像拼接失败。")

        rospy.loginfo("流程完成，节点退出。")


if __name__ == '__main__':
    try:
        node = GoalPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
