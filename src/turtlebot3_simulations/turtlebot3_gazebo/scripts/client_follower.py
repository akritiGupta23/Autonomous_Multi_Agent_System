#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class ClientFollower:
    def __init__(self):
        self.visitor_vel_pub = rospy.Publisher('visitor/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_sub = rospy.Subscriber('client_agent/cmd_vel', Twist, self.cmd_vel_callback)
        self.stop_sub = rospy.Subscriber('stop_follower', Bool, self.stop_callback)
        self.status_pub = rospy.Publisher('follower_status', Bool, queue_size=10)
        self.stop_follower = False
        self.rate = rospy.Rate(20)  # 20 Hz
        rospy.loginfo("Visitor node initialized and listening to client_agent/cmd_vel.")

    def cmd_vel_callback(self, msg: Twist):
        if not self.stop_follower:
            rospy.loginfo("Relaying velocity command from client_agent to visitor.")
            self.visitor_vel_pub.publish(msg)
        else:
            rospy.loginfo("Follower is stopped. Ignoring velocity command.")

    def stop_callback(self, msg: Bool):
        self.stop_follower = msg.data
        status_msg = Bool()
        status_msg.data = not self.stop_follower
        self.status_pub.publish(status_msg)
        rospy.loginfo(f"Follower {'stopped' if self.stop_follower else 'resumed'}.")

    def run(self):
        while not rospy.is_shutdown():
            if not self.stop_follower:
                self.rate.sleep()
            else:
                rospy.loginfo_throttle(10, "Follower is stopped. Waiting for resume command.")
                rospy.sleep(1)

if __name__ == '__main__':
    rospy.init_node('client_follower')
    try:
        follower = ClientFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
