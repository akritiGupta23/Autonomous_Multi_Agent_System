#!/usr/bin/env python3

import rospy
import threading
import numpy as np
from path_planning import NavigationPlanner
from client_follower import ClientFollower
from building_follower import BuildingFollower
from typing import Dict, Tuple

class NavigationSystem:
    def __init__(self, map_path: str):
        self.planner = NavigationPlanner(start_pose=(0.0, 0.0), goal_pose=(10.0, 10.0), map_yaml_path=map_path)
        self.occupancy_grid, _, _ = self.planner.load_map_data()
        self.building_follower = BuildingFollower()
        self.client_follower = ClientFollower()

    def run_agent(self, agent_type: str, path: list):
        if agent_type == "building":
            result = self.planner.move_building_agent(path)
        elif agent_type == "client":
            result = self.planner.move_client_agent(path)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        print(f"{agent_type.capitalize()} agent result:", result)
        if result == "Reached destination":
            self.building_follower.stop_follower = True
            self.client_follower.stop_follower = True

    def start_navigation(self, start_pose: Tuple[float, float], goal_pose: Tuple[float, float]):
        path = self.planner.a_star_pathfinding(self.occupancy_grid, start_pose, goal_pose)
        print("Calculated path:", path)

        building_agent_thread = threading.Thread(target=self.run_agent, args=("building", path))
        building_follower_thread = threading.Thread(target=self.building_follower.run)
        client_follower_thread = threading.Thread(target=self.client_follower.run)

        building_agent_thread.start()
        building_follower_thread.start()
        client_follower_thread.start()

        building_agent_thread.join()
        building_follower_thread.join()
        client_follower_thread.join()

def main():
    rospy.init_node('navigation_system_node', anonymous=True)

    map_path = '/home/akriti/catkin_ws/map.yaml'
    nav_system = NavigationSystem(map_path)

    print("Building agent + visitor system operating")
    print("Occupied cells:", np.count_nonzero(nav_system.occupancy_grid == 100))

    rooms: Dict[str, Tuple[float, float]] = {
        'LAB1': (-4, -5),
        'LAB2': (-3, -4),
        'Office': (-2, -3),
        'Discussion Room': (-1, -2)
    }

    visitor_goal = input("Hey Visitor! I am the Building agent. I will escort you to the desired room. "
                         "Where would you like to go? (LAB1/LAB2/Office/Discussion Room): ")

    if visitor_goal not in rooms:
        print("Invalid room selection. Exiting.")
        return

    start_pose = nav_system.planner.inverse_transform_coordinates(0, 0.5)
    goal_pose = nav_system.planner.inverse_transform_coordinates(*rooms[visitor_goal])

    print("Start Position:", start_pose)
    print("Goal Position:", goal_pose)

    nav_system.start_navigation(start_pose, goal_pose)
    print("Kudos Visitor!! You have reached your final location")

if __name__ == "__main__":
    main()
