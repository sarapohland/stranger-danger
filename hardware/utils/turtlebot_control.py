import rospy
from geometry_msgs.msg import Twist

cmd = Twist()
def cmd_callback(data):
    global cmd
    cmd = data

def main():
    rospy.init_node('controller', anonymous=True)
    rate = rospy.Rate(8) # Hz
    cmd_sub = rospy.Subscriber('turtlebot/command', Twist, cmd_callback)
    cmd_pub = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

    straight_cmd = Twist()
    straight_cmd.linear.x = 0.0
    straight_cmd.angular.z = 0.0

    while not rospy.is_shutdown():
        cmd_pub.publish(cmd)
        rate.sleep()
        straight_cmd.linear.x = cmd.linear.x
        cmd_pub.publish(straight_cmd)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
