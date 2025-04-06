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
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion, quaternion_from_euler

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

    def capture_images(self):
        """
        从内部分别保存的图像数据中获取左右摄像头图像。
        如果图像还未接收到，则等待。
        """
        rate = rospy.Rate(10)
        while self.left_img is None or self.right_img is None:
            rospy.loginfo("等待图像数据...")
            rate.sleep()
        # 返回图像的拷贝，确保后续处理不会受新数据影响
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
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            rospy.logerr("无法加载图片: %s", image_path)
            return None

        height, width = image.shape[:2]
        # 计算裁剪区域
        x_start = int(width * 0.18)
        x_end = int(width * 0.9)
        y_start = int(height * 0.04)
        y_end = int(height * 0.5)
        cropped_image = image[y_start:y_end, x_start:x_end]

        # 定义目标RGB值和允许偏差
        target_colors = [
            np.array([3, 50, 102]),
            np.array([137, 148, 115]),
            np.array([55, 55, 55]),
            np.array([85, 98, 60])
        ]
        tolerance = 10

        # 创建一个白色图像
        filtered_image = np.full_like(cropped_image, 255)

        # 遍历每个目标颜色，应用掩码
        for color in target_colors:
            mask = np.all(np.abs(cropped_image - color) <= tolerance, axis=-1)
            filtered_image[mask] = cropped_image[mask]

        # 转换为灰度图像
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        # 将有颜色的区域标为黑色
        gray_with_black_regions = gray_image.copy()
        gray_with_black_regions[np.any(filtered_image != [255, 255, 255], axis=-1)] = 0

        # 形态学操作（开运算去噪声，闭运算填补孔洞）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        processed_image = cv2.morphologyEx(gray_with_black_regions, cv2.MORPH_OPEN, kernel)
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)

        # 获取黑色区域的坐标
        black_points = np.column_stack(np.where(processed_image == 0))

        # 使用 DBSCAN 聚类
        dbscan = DBSCAN(eps=5, min_samples=80)  # 调整 eps 和 min_samples 参数
        labels = dbscan.fit_predict(black_points)

        # 分别获取两个四边形的点
        cluster_1_points = black_points[labels == 0]
        cluster_2_points = black_points[labels == 1]

        # 计算每个聚类的平均x值
        cluster_1_mean_x = np.mean(cluster_1_points[:, 1])
        cluster_2_mean_x = np.mean(cluster_2_points[:, 1])

        # 如果 cluster_1 的平均x值大于 cluster_2，则交换它们
        if cluster_1_mean_x > cluster_2_mean_x:
            cluster_1_points, cluster_2_points = cluster_2_points, cluster_1_points

        # 重新计算每个四边形的x最大点，以及y最大点
        cluster_1_x_min = cluster_1_points[np.argmin(cluster_1_points[:, 1])]
        cluster_1_x_max = cluster_1_points[np.argmax(cluster_1_points[:, 1])]

        cluster_2_x_max = cluster_2_points[np.argmax(cluster_2_points[:, 1])]
        cluster_2_y_max = cluster_2_points[np.argmax(cluster_2_points[:, 0])]

        # 定义 A_global 和 D_global
        A_global = np.array([9, -23])
        D_global = np.array([9, -1.5])

        # 计算图片坐标系下 AB 和 CD 的长度
        AB_length_image = np.linalg.norm(cluster_1_x_min - cluster_1_x_max)
        CD_length_image = np.linalg.norm(cluster_2_x_max - cluster_2_y_max)

        # 计算全局坐标系下 AD 的长度
        AD_length_global = np.linalg.norm(D_global - A_global)

        # 计算图片坐标系下 AD 的长度
        AD_length_image = np.linalg.norm(cluster_1_x_min - cluster_2_x_max)

        # 计算 B 和 C 在全局坐标系下的修正坐标
        B_global = A_global + 0.55 * (D_global - A_global) * (AB_length_image / AD_length_image)
        C_global = D_global + 1.2 * (A_global - D_global) * (CD_length_image / AD_length_image)

        # 计算 BC 中点在全局坐标系下的坐标
        BC_mid_global = (B_global + C_global) / 2
        rospy.loginfo("计算得到 BC_mid_global: %s", BC_mid_global)
        return BC_mid_global

    def unlock_bridge(self):
        rospy.sleep(1)  # 确保发布者已经注册
        self.bridge_unlock_pub.publish(True)
        rospy.loginfo("Sent unlock command: True")

    def preprocess_image(self, image):
        # 2. 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 4. 固定阈值二值化
        ret, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

        # 5. 使用连通域分析去除小噪点
        min_area = 50  # 根据实际情况调整
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for i in range(1, num_labels):  # 忽略背景标签 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == i] = 255

        # 6. 形态学闭操作填充数字内部的间隙
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        # 7. 反转图像
        processed = cv2.bitwise_not(closed)

        return image, gray, processed

    def ocr_digits(self, processed_image):
        """执行 OCR 识别（使用 pytesseract，仅使用 CPU）"""
        # 使用 pytesseract 执行 OCR 识别，--psm 6 适用于单一文本块
        result = pytesseract.image_to_string(processed_image, config='--psm 6')
        # 合并所有识别结果，并过滤出数字
        digits = ''.join(filter(str.isdigit, result))
        return digits.strip()

    def adjust_speed(self, new_max_vel_x):
        # 替换 'trajectory_planner' 为你的局部规划器节点名称
        client = Client("move_base/TrajectoryPlannerROS", timeout=30)
        params = {'max_vel_x': new_max_vel_x}
        config = client.update_configuration(params)
        rospy.loginfo("Updated max_vel_x to: %s", config['max_vel_x'])

    def run(self):
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

        # 拍摄第一组图像
        rospy.loginfo("拍摄第一组图像...")
        left0, right0 = self.capture_images()
        cv2.imwrite(os.path.join(self.image_folder, "left0.jpg"), left0)
        cv2.imwrite(os.path.join(self.image_folder, "right0.jpg"), right0)
        rospy.loginfo("第一组图像已保存")
        rospy.sleep(1.0)

        # 发布目标使机器人旋转 30°（第一次旋转）
        rospy.loginfo("发布目标点，机器人旋转 30°...")
        new_yaw = self.goal_yaw + math.radians(30)
        self.goal_yaw = new_yaw
        new_quat = self.quaternion_from_yaw(new_yaw)
        self.goal_pose.pose.orientation.x = new_quat[0]
        self.goal_pose.pose.orientation.y = new_quat[1]
        self.goal_pose.pose.orientation.z = new_quat[2]
        self.goal_pose.pose.orientation.w = new_quat[3]
        self.goal_pose.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.goal_pose)
        rospy.sleep(5.0)

        # 拍摄第二组图像
        rospy.loginfo("拍摄第二组图像...")
        left1, right1 = self.capture_images()
        cv2.imwrite(os.path.join(self.image_folder, "left1.jpg"), left1)
        cv2.imwrite(os.path.join(self.image_folder, "right1.jpg"), right1)
        rospy.loginfo("第二组图像已保存")
        rospy.sleep(1.0)

        # 再次发布目标使机器人再旋转 30°（第二次旋转）
        rospy.loginfo("再次发布目标点，机器人旋转 30°...")
        new_yaw = self.goal_yaw + math.radians(30)
        self.goal_yaw = new_yaw
        new_quat = self.quaternion_from_yaw(new_yaw)
        self.goal_pose.pose.orientation.x = new_quat[0]
        self.goal_pose.pose.orientation.y = new_quat[1]
        self.goal_pose.pose.orientation.z = new_quat[2]
        self.goal_pose.pose.orientation.w = new_quat[3]
        self.goal_pose.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.goal_pose)
        rospy.sleep(5.0)

        # 拍摄第三组图像
        rospy.loginfo("拍摄第三组图像...")
        left2, right2 = self.capture_images()
        cv2.imwrite(os.path.join(self.image_folder, "left2.jpg"), left2)
        cv2.imwrite(os.path.join(self.image_folder, "right2.jpg"), right2)
        rospy.loginfo("第三组图像已保存")

        # 拼接六张图像
        rospy.loginfo("开始拼接全景图像...")
        images = [left0, right0, left1, right1, left2, right2]
        stitched = self.stitch_six_images(images)

        if stitched is not None:
            stitched_path = os.path.join(self.image_folder, "p.jpg")
            cv2.imwrite(stitched_path, stitched)
            rospy.loginfo("全景图像已保存为 %s", stitched_path)
            cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
            cv2.imshow("Panorama", stitched)
            cv2.waitKey(1)
        else:
            rospy.logerr("全景图像拼接失败。")
            return

        image_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
        height, width = stitched.shape[:2]
        # 创建一个与 stitched 大小相同的掩码，初始为全白（255）
        mask = np.ones_like(image_rgb, dtype=np.uint8) * 255

        # 定义三角形的三个顶点，示例顶点可根据实际需求进行调整
        triangle = np.array([
            [0, 0],
            [0, int(0.3 * height)],
            [int(0.85 * width), 0]
        ])

        # 在掩码上绘制三角形并填充为黑色 -> 表示需要“抹去”该三角形区域
        cv2.fillPoly(mask, [triangle], (0, 0, 0))

        # 将掩码应用到 stitched 上，三角形区域将被置为 0（即黑色）
        stitched2 = cv2.bitwise_and(stitched, mask)

        # 预处理图像（仅使用 CPU）
        image, gray, processed = self.preprocess_image(stitched2)

        # OCR 识别（使用 pytesseract，仅使用 CPU）
        digits = self.ocr_digits(processed)
        print("最终识别结果:", digits)

        # 利用 YOLO 对全景图像进行目标检测
        rospy.loginfo("开始利用 YOLO 对全景图像进行目标检测...")
        rgb_stitched = cv2.cvtColor(stitched2, cv2.COLOR_BGR2RGB)
        start_time = time()
        results = self.yolo_model(rgb_stitched, show=False, conf=0.3)
        inference_time = time() - start_time
        rospy.loginfo("YOLO 推理时间: {:.3f} s".format(inference_time))
        result_yolo = results[0]
        box_count = len(result_yolo.boxes)
        rospy.loginfo("检测到目标框数量: {}".format(box_count))
        self.detection_result_pub.publish(box_count)
        # 显示检测结果（可选）
        detection_frame = result_yolo.plot()
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", 800, 600)

        cv2.imshow("Detection", detection_frame)

        cv2.waitKey(10000)
        cv2.destroyAllWindows()

        # 调用图像处理函数，根据全景图像计算下一个目标点全局坐标 BC_mid_global
        next_goal = self.compute_next_goal_from_image(stitched_path)
        if next_goal is None:
            rospy.logwarn("未能计算出下一个目标点，流程结束。")
            return

        # 将计算得到的 BC_mid_global 作为下一个目标点（这里只更新 y 坐标），并将 yaw 设置为 180°（即机器人反向）
        rospy.loginfo("更新目标点为 BC_mid_global: %s，yaw 设置为 180°", next_goal)
        self.goal_pose.pose.position.x = 10
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.goal_pose.pose.position.y = next_goal[1]
        new_yaw = math.radians(179)
        new_quat = self.quaternion_from_yaw(new_yaw)
        self.goal_pose.pose.orientation.x = new_quat[0]
        self.goal_pose.pose.orientation.y = new_quat[1]
        self.goal_pose.pose.orientation.z = new_quat[2]
        self.goal_pose.pose.orientation.w = new_quat[3]
        self.goal_pose.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.goal_pose)
        rospy.loginfo("新的目标点已发布，等待到达目标...")

        rospy.sleep(5.0)

        self.reached = False
        # 等待机器人到达目标点（可根据实际情况调整等待机制）
        rate = rospy.Rate(10)  # 10Hz 检查一次
        while not rospy.is_shutdown() and not self.reached:
            self.goal_pose.header.stamp = rospy.Time.now()
            self.goal_pub.publish(self.goal_pose)
            current_time = rospy.Time.now().to_sec()
            if self.last_amcl_time is not None:
                if (current_time - self.last_amcl_time) >= self.stop_duration_required:
                    rospy.loginfo("/amcl_pose 已停止发布 3 秒，进入过桥流程...")
                    self.adjust_speed(20)
                    self.reached = True
            rate.sleep()

        if not self.reached:
            rospy.logwarn("节点被关闭或其它原因终止，未检测到 /amcl_pose 停止。")
            return

        # 解锁桥梁：发布解锁命令
        self.unlock_bridge()

        rospy.loginfo("新的目标点已发布且桥梁已解锁，导航到下一个点。")
        rospy.loginfo("更新目标点为: %s，yaw 设置为 180°", next_goal)
        self.goal_pose.pose.position.x = 3


        #！！！！！！！！！！！！！！！！！！！！！！！
        self.goal_pose.pose.position.y = next_goal[1]
        new_yaw = math.radians(180)
        new_quat = self.quaternion_from_yaw(new_yaw)
        self.goal_pose.pose.orientation.x = new_quat[0]
        self.goal_pose.pose.orientation.y = new_quat[1]
        self.goal_pose.pose.orientation.z = new_quat[2]
        self.goal_pose.pose.orientation.w = new_quat[3]
        self.goal_pose.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.goal_pose)
        rospy.loginfo("新的目标点已发布，等待到达目标...")
        self.unlock_bridge()
        rospy.sleep(1.0)
        self.reached = False
        # 等待机器人到达目标点（可根据实际情况调整等待机制）
        rate = rospy.Rate(10)  # 10Hz 检查一次
        while not rospy.is_shutdown() and not self.reached:
            self.goal_pose.header.stamp = rospy.Time.now()
            self.goal_pub.publish(self.goal_pose)
            current_time = rospy.Time.now().to_sec()
            if self.last_amcl_time is not None:
                if (current_time - self.last_amcl_time) >= self.stop_duration_required:
                    rospy.loginfo("/amcl_pose 已停止发布 3 秒，进入找最终目的地流程...")
                    self.reached = True
            rate.sleep()

        if not self.reached:
            rospy.logwarn("节点被关闭或其它原因终止，未检测到 /amcl_pose 停止。")
            return
        rospy.loginfo("流程完成，节点退出.")


if __name__ == '__main__':
    try:
        node = GoalPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
