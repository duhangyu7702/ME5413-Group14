#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import cv2
import numpy as np
import os
from time import time
from pathlib import Path

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import Image, PointCloud2  # 修改：使用 PointCloud2 替代 LaserScan
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# 新增导入点云处理工具
import sensor_msgs.point_cloud2 as pc2

# 引入 YOLO 库（Ultralytics）
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import pytesseract

# 新增导入 dynamic_reconfigure 客户端
from dynamic_reconfigure.client import Client


class GoalPublisher(object):
    def __init__(self):
        rospy.init_node('goal_publisher', anonymous=True)

        # 发布目标点 -> move_base_simple/goal
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # 发布检测结果（目标框数量）
        self.detection_result_pub = rospy.Publisher('/detection_result', Int32, queue_size=10)

        # 订阅 AMCL 发布的位姿 (PoseWithCovarianceStamped)
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)

        # 发布解锁桥梁消息
        self.bridge_unlock_pub = rospy.Publisher('/cmd_open_bridge', Bool, queue_size=10)

        # 控制机器人旋转 -> /cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # 用于表示障碍物检测状态（检测到障碍物后置为 True）
        self.baffle_stopped = False

        # 修改：订阅三维雷达数据（假设话题为 /radar/points，消息类型 PointCloud2）
        self.radar_sub = rospy.Subscriber('/mid/points', PointCloud2, self.radar_callback)

        # 初始时禁用雷达检测，只有在最后发布 target_goal 后才启用
        self.detection_enabled = False

        # 构造初始目标点消息（坐标和角度）
        self.goal_pose = PoseStamped()
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.pose.position.x = 20.56
        self.goal_pose.pose.position.y = -21.99
        self.goal_pose.pose.orientation.z = 0.30
        self.goal_pose.pose.orientation.w = 0.95

        # 标记：当 /amcl_pose 停止发布 3 秒后进入拍照流程
        self.reached = False

        # 记录最近一次接收 /amcl_pose 的时间和位姿
        self.last_amcl_time = None
        self.current_pose = None
        self.stop_duration_required = 3.0  # 3 秒无消息则认为已停止

        # 用于将 ROS 图像转换为 OpenCV 图像
        self.bridge = CvBridge()

        # 用于保存最新的图像数据（主动订阅图像，不依赖于 rviz）
        self.left_img = None
        self.right_img = None
        self.left_sub = rospy.Subscriber('/left_camera/image_raw', Image, self.left_image_callback)
        self.right_sub = rospy.Subscriber('/left_camera_2/image_raw', Image, self.right_image_callback)

        # 从初始四元数计算当前 yaw 值
        qz = self.goal_pose.pose.orientation.z
        qw = self.goal_pose.pose.orientation.w
        self.goal_yaw = self.quaternion_to_yaw(0.0, 0.0, qz, qw)

        # 初始化 YOLO 模型，用于全景图像的目标检测
        weight_path = "yolov8s.pt"  # 请确保权重文件存在
        rospy.loginfo("加载 YOLO 模型...")
        self.yolo_model = YOLO(weight_path)
        self.yolo_model.fuse()  # 融合模型以提升推理速度

        # 延时 3 秒，确保系统初始化并开始接收 /amcl_pose 消息以及图像数据
        rospy.loginfo("等待系统初始化 3 秒...")
        rospy.sleep(3.0)
        self.last_amcl_time = rospy.Time.now().to_sec()
        rospy.loginfo("系统初始化完成，开始正常运行。")

        # 设置图片保存的文件夹：上一级目录的 image 文件夹
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        self.image_folder = os.path.join(parent_dir, "image")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
            rospy.loginfo("创建图片保存文件夹: %s", self.image_folder)

    def left_image_callback(self, img_msg):
        """左摄像头图像回调函数，保存最新图像"""
        self.left_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def right_image_callback(self, img_msg):
        """右摄像头图像回调函数，保存最新图像"""
        self.right_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """将四元数转换为欧拉角，然后返回 yaw。"""
        (roll, pitch, yaw) = euler_from_quaternion([qx, qy, qz, qw])
        return yaw

    def quaternion_from_yaw(self, yaw, roll=0.0, pitch=0.0):
        """将 yaw 转换为四元数（假设 roll 和 pitch 为 0）。"""
        return quaternion_from_euler(roll, pitch, yaw)

    def pose_callback(self, pose_msg):
        """回调函数：更新最近一次接收 /amcl_pose 的时间和当前位姿。"""
        self.last_amcl_time = rospy.Time.now().to_sec()
        self.current_pose = pose_msg.pose.pose

    def radar_callback(self, point_cloud_msg):
        """
        三维雷达回调函数：检测小车左后方的障碍物。
        只关注角度在110°到120°之间且高度在0.5m ± 0.1m范围内的点，
        当检测到距离低于阈值（例如3.0米）时，发布停车指令。
        """
        if not self.detection_enabled:
            return

        threshold_distance = 3.0  # 距离阈值（单位：米）
        tolerance_z = 0.1  # 高度容差：0.5m ± 0.1m

        # 定义检测角度范围：110°到120°（转换为弧度）
        angle_min = math.radians(110)
        angle_max = math.radians(120)

        valid_ranges = []

        # 遍历点云中的每个点（读取 x, y, z 数据），跳过无效点
        for point in pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            # 计算相对于 x 轴的夹角，返回范围为 [-pi, pi]
            angle = math.atan2(y, x)
            # 将负角度转换到 [0, 2pi] 范围
            if angle < 0:
                angle += 2 * math.pi

            # 判断点是否在指定角度范围内，且高度在0.5m ± 0.1m范围内
            if angle_min <= angle <= angle_max and abs(z - 0.5) < tolerance_z:
                distance = math.hypot(x, y)
                valid_ranges.append(distance)

        if valid_ranges:
            min_range = min(valid_ranges)
            if min_range < threshold_distance and not self.baffle_stopped:
                self.baffle_stopped = True
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self.cmd_vel_pub.publish(stop_twist)
                rospy.loginfo("检测到左后方角度在110°到120°且高度0.5m附近障碍物（距离：%.2f m），发布停车指令", min_range)



    def capture_images(self):
        """
        从内部分别保存的图像数据中获取左右摄像头图像。
        如果图像还未接收到，则等待。
        """
        rate = rospy.Rate(10)
        while self.left_img is None or self.right_img is None:
            rospy.loginfo("等待图像数据...")
            rate.sleep()
        return self.left_img.copy(), self.right_img.copy()

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

    def compute_next_goal_from_image(self, image_path):
        """
        封装原有图像处理代码：读取 image_path 指定的图片，进行裁剪、颜色过滤、形态学处理、DBSCAN 聚类，
        并根据计算得到 B_global 和 C_global 最后求得 BC_mid_global，作为下一个目标点的全局坐标。
        """
        image = cv2.imread(image_path)
        if image is None:
            rospy.logerr("无法加载图片: %s", image_path)
            return None

        height, width = image.shape[:2]
        x_start = int(width * 0.18)
        x_end = int(width * 0.9)
        y_start = int(height * 0.04)
        y_end = int(height * 0.5)
        cropped_image = image[y_start:y_end, x_start:x_end]

        target_colors = [
            np.array([3, 50, 102]),
            np.array([137, 148, 115]),
            np.array([55, 55, 55]),
            np.array([85, 98, 60])
        ]
        tolerance = 10
        filtered_image = np.full_like(cropped_image, 255)
        for color in target_colors:
            mask = np.all(np.abs(cropped_image - color) <= tolerance, axis=-1)
            filtered_image[mask] = cropped_image[mask]

        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        gray_with_black_regions = gray_image.copy()
        gray_with_black_regions[np.any(filtered_image != [255, 255, 255], axis=-1)] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        processed_image = cv2.morphologyEx(gray_with_black_regions, cv2.MORPH_OPEN, kernel)
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)

        black_points = np.column_stack(np.where(processed_image == 0))
        dbscan = DBSCAN(eps=5, min_samples=80)
        labels = dbscan.fit_predict(black_points)

        cluster_1_points = black_points[labels == 0]
        cluster_2_points = black_points[labels == 1]

        cluster_1_mean_x = np.mean(cluster_1_points[:, 1])
        cluster_2_mean_x = np.mean(cluster_2_points[:, 1])
        if cluster_1_mean_x > cluster_2_mean_x:
            cluster_1_points, cluster_2_points = cluster_2_points, cluster_1_points

        cluster_1_x_min = cluster_1_points[np.argmin(cluster_1_points[:, 1])]
        cluster_1_x_max = cluster_1_points[np.argmax(cluster_1_points[:, 1])]

        cluster_2_x_max = cluster_2_points[np.argmax(cluster_2_points[:, 1])]
        cluster_2_y_max = cluster_2_points[np.argmax(cluster_2_points[:, 0])]

        A_global = np.array([9, -23])
        D_global = np.array([9, -1.5])

        AB_length_image = np.linalg.norm(cluster_1_x_min - cluster_1_x_max)
        CD_length_image = np.linalg.norm(cluster_2_x_max - cluster_2_y_max)
        AD_length_global = np.linalg.norm(D_global - A_global)
        AD_length_image = np.linalg.norm(cluster_1_x_min - cluster_2_x_max)

        B_global = A_global + 0.55 * (D_global - A_global) * (AB_length_image / AD_length_image)
        C_global = D_global + 1.2 * (A_global - D_global) * (CD_length_image / AD_length_image)

        BC_mid_global = (B_global + C_global) / 2
        rospy.loginfo("计算得到 BC_mid_global: %s", BC_mid_global)
        return BC_mid_global

    def unlock_bridge(self):
        rospy.sleep(1)
        self.bridge_unlock_pub.publish(True)
        rospy.loginfo("Sent unlock command: True")

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
        min_area = 50
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == i] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        processed = cv2.bitwise_not(closed)
        return image, gray, processed

    def ocr_digits(self, processed_image):
        result = pytesseract.image_to_string(processed_image, config='--psm 6')
        digits = ''.join(filter(str.isdigit, result))
        return digits.strip()

    def adjust_speed(self, new_max_vel_x):
        client = Client("move_base/TrajectoryPlannerROS", timeout=30)
        params = {'max_vel_x': new_max_vel_x}
        config = client.update_configuration(params)
        rospy.loginfo("Updated max_vel_x to: %s", config['max_vel_x'])

    def drive_along_river_edge(self, forward_speed=0.5):
        """
        沿河边行走过程中直接发布速度控制命令
        :param forward_speed: 前进速度，可根据实际需求调整
        """
        rospy.loginfo("开始沿河边行走：直接通过 cmd_vel 控制前进")
        # 启用雷达检测（确保 radar_callback 能正常判断障碍物）
        self.detection_enabled = True
        rate = rospy.Rate(10)  # 控制循环频率为10Hz
        forward_twist = Twist()
        forward_twist.linear.x = forward_speed  # 设置前进速度
        forward_twist.angular.z = 0.0  # 保持直行

        # 当未检测到障碍物时持续前进
        while not rospy.is_shutdown() and not self.baffle_stopped:
            self.cmd_vel_pub.publish(forward_twist)
            rate.sleep()

        # 一旦检测到障碍物或满足终止条件，发送零速命令停车
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        rospy.loginfo("沿河边行走过程中检测到障碍物，车辆已停止。")
        # 可在此处关闭雷达检测标志，若后续流程需要
        self.detection_enabled = False

    def run(self):


        self.drive_along_river_edge(forward_speed=0.5)

if __name__ == '__main__':
    try:
        node = GoalPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
