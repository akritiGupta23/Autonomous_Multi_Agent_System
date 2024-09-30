#!/usr/bin/env python3

import rospy
from path_planning import PathPlanningTool
from client_follower import Client_Follower
from building_follower import Client_building_Follower
import threading
import numpy as np
from utils import *


def run_client_agent(obj, path, follower):
    result = obj.move_client_agent(path)
    print(result)
    if result == "Reached destination":
        follower.stop_follower = True

def run_building_agent(obj, path, follower, follower2):
    result = obj.move_building_agent(path)
    print(result)
    if result == "Reached destination":
        follower.stop_follower = True
        follower2.stop_follower = True

def main():
    rospy.init_node('path_planning_node', anonymous=True)
    
    obj = PathPlanningTool(start_pose=(0.0, 0.0), goal_pose=(10.0, 10.0), map_yaml_path='/home/akriti/catkin_ws/map.yaml')
    occupancy_grid, _, _ = obj.load_map()
    print(1)
    print(np.count_nonzero(occupancy_grid == 100))

    print("Building agent + visitor system operating")
    start_pose = obj.inverse_convert_coordinates(0, 0.5)

    room = {'LAB1': (-4,-5)}
    visitor_goal = input("Hey Visitor!! I am Building agent. I will escort you to the desired room. You want to go to LAB1 or LAB2 or Office or Discussion Room")

    goal_pose = obj.inverse_convert_coordinates(room[visitor_goal][0], room[visitor_goal][1])

    print("Start Position : ", start_pose)
    print("Goal Position : ", goal_pose)

    path = obj.a_star_algorithm(occupancy_grid, start_pose, goal_pose)
    print(path)

    follower = Client_building_Follower()
    follower2 = Client_Follower()

    building_agent_thread = threading.Thread(target=run_building_agent, args=(obj, path, follower, follower2))
    building_agent_follower_thread = threading.Thread(target=follower.run)
    client_follow_builder = threading.Thread(target=follower2.run)

    building_agent_thread.start()
    building_agent_follower_thread.start()
    client_follow_builder.start()

    building_agent_thread.join()
    building_agent_follower_thread.join()
    client_follow_builder.join()

    print("Kudos Visitor!! You have reached your final location")

if __name__ == "__main__":
    main()

