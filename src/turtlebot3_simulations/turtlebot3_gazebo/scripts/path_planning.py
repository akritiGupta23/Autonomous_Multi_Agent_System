#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Optional, Tuple, List
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import heapq
from collections import deque
import math
from tf.transformations import euler_from_quaternion
from utils import *

class NavigationPlanner(BaseModel):
    initial_position: Optional[Tuple[float, float]] = None
    target_position: Optional[Tuple[float, float]] = None
    map_file_path: Optional[str] = None
    
    grid_map: Optional[np.ndarray] = None
    cell_resolution: Optional[float] = None
    grid_origin: Optional[Tuple[float, float, float]] = None
    agent_positions: dict = {}
    agent_orientations: dict = {}
    
    class Config:
        arbitrary_types_allowed = True

    def load_map_data(self):
        map_contents = read_yaml_map_file(self.map_file_path)
        self.grid_map, self.cell_resolution, self.grid_origin = convert_map_image_to_grid(map_contents)
        return self.grid_map, self.cell_resolution, self.grid_origin


    @staticmethod
    def a_star_pathfinding(grid_map: np.ndarray, start_point: Tuple[int, int], target_point: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start_point, target_point = map(lambda x: tuple(map(round, x)), (start_point, target_point))
        if grid_map[start_point] != 100:
            print("Invalid starting point")
            return None

        def heuristic_function(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        priority_queue = [(0, start_point, [start_point])]
        visited_nodes = set([start_point])
        movement_directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        steps_counter = 0

        while priority_queue:
            steps_counter += 1
            total_cost, current_node, path_traced = heapq.heappop(priority_queue)

            if current_node == target_point:
                print(f"Path found in {steps_counter} steps")
                return path_traced
            
            for dx, dy in movement_directions:
                new_node = (current_node[0] + dx, current_node[1] + dy)
                if new_node not in visited_nodes and grid_map[new_node] == 100:
                    new_cost = total_cost + math.hypot(dx, dy) + heuristic_function(new_node, target_point)
                    heapq.heappush(priority_queue, (new_cost, new_node, path_traced + [new_node]))
                    visited_nodes.add(new_node)
        
        print("Path not found")
        return None
    
    def odometry_callback(self, msg, agent_name):
        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.agent_positions[agent_name] = position
        
        orientation_quat = msg.pose.pose.orientation
        orientation_list = [orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w]
        _, _, agent_orientation = euler_from_quaternion(orientation_list)
        self.agent_orientations[agent_name] = agent_orientation

    def navigate_agent(self, agent_name: str, path: List[Tuple[int, int]]):
        odom_subscriber = rospy.Subscriber(f"{agent_name}/odom", Odometry, lambda msg: self.odometry_callback(msg, agent_name))
        velocity_publisher = rospy.Publisher(f'{agent_name}/cmd_vel', Twist, queue_size=10)
        velocity_message = Twist()
        rate = rospy.Rate(10)
        
        while agent_name not in self.agent_positions:
            rospy.loginfo("Waiting for odometry data...")
            rate.sleep()

        rospy.loginfo("Odometry data received, starting movement...")

        for current_position, next_position in zip(path, path[1:]):
            current_position = transform_coordinates(*current_position)
            next_position = transform_coordinates(*next_position)

            while not self.has_reached_point(agent_name, next_position):
                current_position_data = self.agent_positions[agent_name]
                current_orientation_data = self.agent_orientations[agent_name]
                
                target_angle = math.atan2(next_position[1] - current_position_data[1], 
                                          next_position[0] - current_position_data[0])
                angle_difference = NavigationPlanner.normalize_angle(target_angle - current_orientation_data)

                if abs(angle_difference) > 0.1:
                    velocity_message.linear.x = 0.0
                    velocity_message.angular.z = 0.2 * angle_difference
                else:
                    velocity_message.angular.z = 0.0
                    velocity_message.linear.x = 0.3 if agent_name != "building_agent" else 0.5

                velocity_publisher.publish(velocity_message)
                rate.sleep()

            velocity_message.linear.x = 0.0
            velocity_message.angular.z = 0.0
            velocity_publisher.publish(velocity_message)
            rate.sleep()

        velocity_message.linear.x = 0.0
        velocity_message.angular.z = 0.0
        velocity_publisher.publish(velocity_message)
        
        return "Reached destination"

    def has_reached_point(self, agent_name: str, target_point: Tuple[float, float], tolerance: float = 0.1) -> bool:
        if agent_name not in self.agent_positions:
            return False
        current_position_data = self.agent_positions[agent_name]
        distance = math.hypot(target_point[0] - current_position_data[0], target_point[1] - current_position_data[1])
        return distance < tolerance

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
        
    def reverse_transform_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        return inverse_transform_coordinates(x, y)

