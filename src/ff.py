#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import cv2
import numpy as np
import os
from time import time
from pathlib import Path
from collections import Counter  # Additional import for counting occurrences of numbers

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Import YOLO library (Ultralytics)
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import pytesseract

# Additional import for dynamic_reconfigure client
from dynamic_reconfigure.client import Client
from gazebo_msgs.msg import ModelStates

class GoalPublisher(object):
    def __init__(self):
        rospy.init_node('goal_publisher', anonymous=True)

        # Publish target goal -> move_base_simple/goal
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # Publish detection result (number of target boxes)
        self.detection_result_pub = rospy.Publisher('/detection_result', Int32, queue_size=10)

        # Subscribe to AMCL published pose (PoseWithCovarianceStamped)
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)

        # Publish bridge unlock message
        self.bridge_unlock_pub = rospy.Publisher('/cmd_open_bridge', Bool, queue_size=10)

        # Control robot rotation -> /cmd_vel (in this example, rotation is controlled mainly by publishing target goals)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Construct initial target goal message (coordinates and angle)
        self.goal_pose = PoseStamped()
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.pose.position.x = 20.56
        self.goal_pose.pose.position.y = -21.99
        # The initial angle's corresponding quaternion (only z and w are set, other components are 0)
        self.goal_pose.pose.orientation.z = 0.30
        self.goal_pose.pose.orientation.w = 0.95

        # Flag: Enter the image capture process when /amcl_pose stops publishing for 3 seconds
        self.reached = False

        # Record the time and pose of the most recent /amcl_pose message
        self.last_amcl_time = None
        self.current_pose = None
        self.stop_duration_required = 3.0  # 3 seconds without a message is considered stopped

        # Used to convert ROS images to OpenCV images
        self.bridge = CvBridge()

        # Used to store the latest image data (actively subscribed to images, not dependent on rviz)
        self.left_img = None
        self.right_img = None
        self.left_sub = rospy.Subscriber('/left_camera/image_raw', Image, self.left_image_callback)
        self.right_sub = rospy.Subscriber('/left_camera_2/image_raw', Image, self.right_image_callback)

        # Calculate the current yaw value from the initial quaternion
        qz = self.goal_pose.pose.orientation.z
        qw = self.goal_pose.pose.orientation.w
        self.goal_yaw = self.quaternion_to_yaw(0.0, 0.0, qz, qw)

        # Initialize YOLO model for target detection in panoramic images
        weight_path = "yolov8s.pt"  # Please ensure the weight file exists
        rospy.loginfo("Loading YOLO model...")
        self.yolo_model = YOLO(weight_path)
        self.yolo_model.fuse()  # Fuse the model to improve inference speed

        # Delay for 3 seconds to ensure system initialization and start receiving /amcl_pose messages and image data
        rospy.loginfo("Waiting for system initialization for 3 seconds...")
        rospy.sleep(3.0)
        self.last_amcl_time = rospy.Time.now().to_sec()
        rospy.loginfo("System initialization complete, starting normal operation.")

        # Set the image saving folder: "image" folder in the parent directory
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        self.image_folder = os.path.join(parent_dir, "image")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
            rospy.loginfo("Created image saving folder: %s", self.image_folder)

    def left_image_callback(self, img_msg):
        """Left camera image callback function, saving the latest image"""
        self.left_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def right_image_callback(self, img_msg):
        """Right camera image callback function, saving the latest image"""
        self.right_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """Convert quaternion to Euler angles, then return yaw."""
        (roll, pitch, yaw) = euler_from_quaternion([qx, qy, qz, qw])
        return yaw

    def quaternion_from_yaw(self, yaw, roll=0.0, pitch=0.0):
        """Convert yaw to quaternion (assuming roll and pitch are 0)."""
        return quaternion_from_euler(roll, pitch, yaw)

    def pose_callback(self, pose_msg):
        """Callback function: Update the most recent reception time and current pose from /amcl_pose."""
        self.last_amcl_time = rospy.Time.now().to_sec()
        self.current_pose = pose_msg.pose.pose

    def capture_images(self):
        """
        Obtain the left and right camera images from internally stored image data.
        If images have not yet been received, wait.
        """
        rate = rospy.Rate(10)
        while self.left_img is None or self.right_img is None:
            rospy.loginfo("Waiting for image data...")
            rate.sleep()
        # Return a copy of the images to ensure subsequent processing is not affected by new data
        return self.left_img.copy(), self.right_img.copy()

    def stitch_six_images(self, images):
        """
        Use OpenCV's stitcher to perform panoramic stitching on six images.
        Stitching order: [left0, right0, left1, right1, left2, right2]
        """
        stitcher = cv2.Stitcher_create()
        status, stitched = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            rospy.loginfo("Panoramic image stitching succeeded")
            return stitched
        else:
            rospy.logerr("Panoramic image stitching failed, error code: %d", status)
            return None

    def compute_next_goal_from_image(self, image_path):
        """
        Encapsulate the original image processing code: read the image specified by image_path,
        perform cropping, color filtering, morphological processing, and DBSCAN clustering,
        then compute B_global and C_global to finally obtain BC_mid_global as the global coordinates
        for the next target goal.
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            rospy.logerr("Unable to load image: %s", image_path)
            return None

        height, width = image.shape[:2]
        # Calculate cropping region
        x_start = int(width * 0.18)
        x_end = int(width * 0.9)
        y_start = int(height * 0.04)
        y_end = int(height * 0.5)
        cropped_image = image[y_start:y_end, x_start:x_end]

        # Define target RGB values and allowed deviation
        target_colors = [
            np.array([3, 50, 102]),
            np.array([137, 148, 115]),
            np.array([55, 55, 55]),
            np.array([85, 98, 60])
        ]
        tolerance = 10

        # Create a white image
        filtered_image = np.full_like(cropped_image, 255)

        # For each target color, apply mask
        for color in target_colors:
            mask = np.all(np.abs(cropped_image - color) <= tolerance, axis=-1)
            filtered_image[mask] = cropped_image[mask]

        # Convert to grayscale image
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        # Mark colored regions as black
        gray_with_black_regions = gray_image.copy()
        gray_with_black_regions[np.any(filtered_image != [255, 255, 255], axis=-1)] = 0

        # Morphological operations (opening to remove noise, closing to fill holes)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        processed_image = cv2.morphologyEx(gray_with_black_regions, cv2.MORPH_OPEN, kernel)
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)

        # Get coordinates of black regions
        black_points = np.column_stack(np.where(processed_image == 0))

        # Use DBSCAN clustering
        dbscan = DBSCAN(eps=5, min_samples=80)  # Adjust eps and min_samples parameters
        labels = dbscan.fit_predict(black_points)

        # Get points of the two quadrilaterals respectively
        cluster_1_points = black_points[labels == 0]
        cluster_2_points = black_points[labels == 1]

        # Calculate the average x value for each cluster
        cluster_1_mean_x = np.mean(cluster_1_points[:, 1])
        cluster_2_mean_x = np.mean(cluster_2_points[:, 1])

        # If the average x value of cluster_1 is greater than that of cluster_2, swap them
        if cluster_1_mean_x > cluster_2_mean_x:
            cluster_1_points, cluster_2_points = cluster_2_points, cluster_1_points

        # Recalculate the extreme points of each quadrilateral: for cluster_1, min and max x; for cluster_2, max x and max y
        cluster_1_x_min = cluster_1_points[np.argmin(cluster_1_points[:, 1])]
        cluster_1_x_max = cluster_1_points[np.argmax(cluster_1_points[:, 1])]

        cluster_2_x_max = cluster_2_points[np.argmax(cluster_2_points[:, 1])]
        cluster_2_y_max = cluster_2_points[np.argmax(cluster_2_points[:, 0])]

        # Define A_global and D_global
        A_global = np.array([9, -23])
        D_global = np.array([9, -1.5])

        # Calculate the lengths of AB and CD in the image coordinate system
        AB_length_image = np.linalg.norm(cluster_1_x_min - cluster_1_x_max)
        CD_length_image = np.linalg.norm(cluster_2_x_max - cluster_2_y_max)

        # Calculate the length of AD in the global coordinate system
        AD_length_global = np.linalg.norm(D_global - A_global)

        # Calculate the length of AD in the image coordinate system
        AD_length_image = np.linalg.norm(cluster_1_x_min - cluster_2_x_max)

        # Calculate the corrected global coordinates for B and C
        B_global = A_global + 0.55 * (D_global - A_global) * (AB_length_image / AD_length_image)
        C_global = D_global + 1.2 * (A_global - D_global) * (CD_length_image / AD_length_image)

        # Calculate the global coordinates of the midpoint of BC
        BC_mid_global = (B_global + C_global) / 2
        rospy.loginfo("Computed BC_mid_global: %s", BC_mid_global)
        return BC_mid_global

    def unlock_bridge(self):
        rospy.sleep(1)  # Ensure the publisher has been registered
        self.bridge_unlock_pub.publish(True)
        rospy.loginfo("Sent unlock command: True")

    def preprocess_image(self, image):
        # 2. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 4. Fixed thresholding for binarization
        ret, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

        # 5. Remove small noise using connected component analysis
        min_area = 50  # Adjust according to actual situation
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for i in range(1, num_labels):  # Ignore background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == i] = 255

        # 6. Morphological closing to fill gaps inside digits
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        # 7. Invert the image
        processed = cv2.bitwise_not(closed)

        return image, gray, processed

    def ocr_digits(self, processed_image):
        """Perform OCR recognition (using pytesseract, CPU only)"""
        # Use pytesseract to perform OCR recognition, --psm 6 is suitable for a single block of text
        result = pytesseract.image_to_string(processed_image, config='--psm 6')
        # Combine all recognition results and filter out digits
        digits = ''.join(filter(str.isdigit, result))
        return digits.strip()

    def adjust_speed(self, new_max_vel_x):
        # Replace 'trajectory_planner' with your local planner node's name
        client = Client("move_base/TrajectoryPlannerROS", timeout=30)
        params = {'max_vel_x': new_max_vel_x}
        config = client.update_configuration(params)
        rospy.loginfo("Updated max_vel_x to: %s", config['max_vel_x'])

    def get_bridge_global_pose(self, timeout=5.0):
        """
        Wait for the /gazebo/model_states message, find the model with the name "bridge",
        and return its global coordinates (x, y); if it times out or is not found, return (None, None).
        """
        try:
            model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=timeout)
            # Find the index of the model with name "bridge"
            index = model_states.name.index("bridge")
            bridge_pose = model_states.pose[index]
            bridge_y = -bridge_pose.position.x
            bridge_x = bridge_pose.position.y
            rospy.loginfo("Bridge global coordinates: x=%.2f, y=%.2f", bridge_x, bridge_y)
            return bridge_x, bridge_y
        except Exception as e:
            rospy.logerr("Failed to get bridge global coordinates: %s", e)
            return None, None

    def run(self):
        self.unlock_bridge()
        rate = rospy.Rate(10)  # 控制循环频率为10Hz
        forward_twist = Twist()
        forward_twist.linear.x = 8  # 设置车辆前进速度（单位：m/s）
        forward_twist.angular.z = 0.0  # 保持直行

        # 计算前进所需时间，公式：时间 = 距离 / 速度
        travel_time = 2.5  # 此处为 5/5 = 1 秒

        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < travel_time:
            self.cmd_vel_pub.publish(forward_twist)
            rate.sleep()

        # 前进完成后，发布停止指令
        stop_twist = Twist()  # 默认所有参数为0，即停止运动
        self.cmd_vel_pub.publish(stop_twist)
        rospy.loginfo("车辆已前进5米并停止。")

if __name__ == '__main__':
    try:
        node = GoalPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
