import rospy
from sensor_msgs.msg import Image

def image_callback(msg):
    rospy.loginfo("Image width: %d, height: %d", msg.width, msg.height)

rospy.init_node('image_size_listener')
rospy.Subscriber('/left_camera/image_raw', Image, image_callback)
rospy.spin()
