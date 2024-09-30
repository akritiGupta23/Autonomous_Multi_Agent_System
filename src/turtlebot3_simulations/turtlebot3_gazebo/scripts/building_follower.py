#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Optional
import rospy
from geometry_msgs.msg import Twist

class Client_building_Follower(BaseModel):
    visitor_vel_pub: Optional[rospy.Publisher] = None
    cmd_vel_sub: Optional[rospy.Subscriber] = None
    stop_follower: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.cmd_vel_sub = rospy.Subscriber('building_agent/cmd_vel', Twist, self.callback)
        self.visitor_vel_pub = rospy.Publisher('client_agent/cmd_vel', Twist, queue_size=10)
        rospy.loginfo("Visitor node initialized and listening to building_agent/cmd_vel.")

    def callback(self, msg: Twist):
        if not self.stop_follower:
            self.visitor_vel_pub.publish(msg)
        
    def run(self):
        rate = rospy.Rate(20)
        while not self.stop_follower:
            rate.sleep()
        rospy.loginfo("Building follower thread stopped.")

