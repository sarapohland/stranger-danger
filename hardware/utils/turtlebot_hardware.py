import os
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from kobuki_msgs.msg import BumperEvent
from kobuki_msgs.msg import WheelDropEvent
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError


class TurtlebotHardware():

    def __init__(self, params):
        self.params = params

        # Current state of robot
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.angle = 0

        # Tracked states of robot
        self.track_states = False
        self.measured_positions = []
        self.measured_velocities = []
        self.measured_angles = []

        # Collision detection
        self.hit_obstacle = False

        # RGB and depth images
        self.rgb_raw_image = None
        self.depth_raw_image = None

        # Initialize ROS node
        rospy.init_node('Turtlebot_Agent')

        # Initialize Subscribers
        self.odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.bumper = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bump_callback)
        self.wheel_drop = rospy.Subscriber('/mobile_base/events/wheel_drop', WheelDropEvent,  self.wheel_drop_callback)

        self.bridge = CvBridge()
        self.rgb_imager = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_imager_callback)
        self.depth_imager = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_imager_callback)

        # Initialize Publishers
        self.odom_reset = rospy.Publisher('/mobile_base/commands/reset_odometry', Empty, queue_size=5)
        self.cmd_vel = rospy.Publisher('turtlebot/command', Twist, queue_size=10)

        # Initialize rospy
        rospy.sleep(1)
        self.r = rospy.Rate(int(1./params.dt))  # Set the actuator frequency in Hz
        self.reset_odom()

    # Odometry callback
    def odom_callback(self, data):
        quaternion = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        self.angle = euler_from_quaternion(quaternion)[2]
        self.position[0] = data.pose.pose.position.x
        self.position[1] = data.pose.pose.position.y
        self.velocity[0] = data.twist.twist.linear.x * np.cos(self.angle)
        self.velocity[1] = data.twist.twist.linear.x * np.sin(self.angle)
        if self.track_states:
            self.measured_positions.append(1.*np.array(self.position))
            self.measured_velocities.append(1.*np.array(self.velocity))
            self.measured_angles.append(1.*self.angle)

    # RGB camera callback
    def rgb_imager_callback(self, data):
        self.rgb_raw_image = self.bridge.imgmsg_to_cv2(data)

    # Depth camera callback
    def depth_imager_callback(self, data):
        self.depth_raw_image = self.bridge.imgmsg_to_cv2(data)

    # Wheel drop callback
    def wheel_drop_callback(self, data):
        self.hit_obstacle = True

    # Bump detection callback
    def bump_callback(self, data):
        self.hit_obstacle = True

    # Reset odometry to zero position
    def reset_odom(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        while np.linalg.norm(self.velocity) > .001:
            self.cmd_vel.publish(cmd)
            self.r.sleep()

        # Reset the odometer to read [0, 0, 0]
        self.odom_reset.publish(Empty())
        rospy.sleep(1)

    # Send velocity command
    def apply_command(self, u):
        cmd = Twist()
        if not self.hit_obstacle:
            cmd.linear.x = u[0]
            cmd.angular.z = u[1]
        else:
            cmd.linear.x = 0.0
            cmd.linear.z = 0.0
        self.cmd_vel.publish(cmd)
