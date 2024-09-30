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

class PathPlanningTool(BaseModel):
    start_pose: Optional[Tuple[float, float]] = None
    goal_pose: Optional[Tuple[float, float]] = None
    map_yaml_path: Optional[str] = None
    
    occupancy_grid: Optional[np.ndarray] = None
    resolution: Optional[float] = None
    origin: Optional[Tuple[float, float, float]] = None
    current_position_client: Optional[Tuple[float, float]] = None
    current_orientation_client: Optional[float] = None
    current_position_building_agent: Optional[Tuple[float, float]] = None
    current_orientation_building_agent: Optional[float] = None
    current_position_visitor: Optional[Tuple[float, float]] = None
    current_orientation_visitor: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

    def load_map(self):
        map_data = load_map_yaml(self.map_yaml_path)
        self.occupancy_grid, self.resolution, self.origin = map_image_to_grid(map_data)
        return self.occupancy_grid, self.resolution, self.origin

    @staticmethod
    def bfs_traversal(occupancy_grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        q = deque()
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        q.append([start, [start]])

        move_directions: List[Tuple[float, float]] = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        if occupancy_grid[start[0]][start[1]] != 100:
            print("Start point invalid")
            return None
        visited = set()
        visited.add(start)
        while q:
            node, path = q.popleft()

            if node == goal:
                print("path found")
                return path
            
            for dir in move_directions:
                x, y = dir
                newNode = (node[0] + x, node[1] + y)
                if newNode not in visited and occupancy_grid[node[0] + x][node[1] + y] == 100:
                    q.append([newNode, path + [newNode]])
                    visited.add(newNode)
        print("path not found")
        return None

    @staticmethod
    def a_star_algorithm(occupancy_grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start = (round(start[0]), round(start[1]))
        goal = (round(goal[0]), round(goal[1]))
        
        def euclidean_dist(start, end):
            return ((start[0] - end[0])**2 + (start[1] - end[1])**2) ** 0.5
        
        def heuristic_function(start, end):
            return abs(start[0] - end[0]) + abs(start[1] - end[1])

        pq = []
        heapq.heappush(pq, (0, start, [start]))
        move_directions: List[Tuple[float, float]] = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        if occupancy_grid[start[0]][start[1]] != 100:
            print("Start point invalid")
            return None
        visited = set()
        visited.add(start)
        steps = 0
        while pq:
            steps += 1
            cost, node, path = heapq.heappop(pq)

            if node == goal:
                print(f"path found in {steps} steps")
                return path
            
            for dir in move_directions:
                x, y = dir
                newNode = (node[0] + x, node[1] + y)
                if newNode not in visited and occupancy_grid[node[0] + x][node[1] + y] == 100:
                    h = heuristic_function(newNode, goal)
                    heapq.heappush(pq, (cost + euclidean_dist(node, newNode) + h, newNode, path + [newNode]))
                    visited.add(newNode)
        print("path not found")
        return None
    
    def odom_callback_client_agent(self, msg):
        self.current_position_client = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_orientation_client = euler_from_quaternion(orientation_list)
        
    def odom_callback_building_agent(self, msg):
        self.current_position_building_agent = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_orientation_building_agent = euler_from_quaternion(orientation_list)
        
    def odom_callback_visitor(self, msg):
        self.current_position_visitor = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_orientation_visitor = euler_from_quaternion(orientation_list)

    def move_client_agent(self, path: List[Tuple[int, int]]):
        odom_sub = rospy.Subscriber("client_agent/odom", Odometry, self.odom_callback_client_agent)
        vel_pub = rospy.Publisher('client_agent/cmd_vel', Twist, queue_size=10)
        vel_msg = Twist()
        rate = rospy.Rate(10)
        
        while self.current_position_client is None:
            rospy.loginfo("Waiting for odometry data...")
            rate.sleep()

        rospy.loginfo("Odometry data received, starting movement...")

        for i in range(len(path) - 1):
            current_point = convert_coordinates(path[i][0], path[i][1])
            next_point = convert_coordinates(path[i+1][0], path[i+1][1])

            while not self.reached_point_client(next_point):
                target_angle = math.atan2(next_point[1] - self.current_position_client[1], 
                                          next_point[0] - self.current_position_client[0])
                angle_diff = self.normalize_angle(target_angle - self.current_orientation_client)

                if abs(angle_diff) > 0.1:
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.2 * angle_diff
                else:
                    vel_msg.angular.z = 0.0
                    vel_msg.linear.x = 0.3

                vel_pub.publish(vel_msg)
                rate.sleep()

            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            vel_pub.publish(vel_msg)
            rate.sleep()

        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        vel_pub.publish(vel_msg)
        
        return "Reached destination"
        
    def move_building_agent(self, path: List[Tuple[int, int]]):
        odom_sub = rospy.Subscriber("building_agent/odom", Odometry, self.odom_callback_building_agent)
        vel_pub = rospy.Publisher('building_agent/cmd_vel', Twist, queue_size=10)
        vel_msg = Twist()
        rate = rospy.Rate(10)
        
        while self.current_position_building_agent is None:
            rospy.loginfo("Waiting for odometry data...")
            rate.sleep()

        rospy.loginfo("Odometry data received, starting movement...")

        for i in range(len(path) - 1):
            current_point = convert_coordinates(path[i][0], path[i][1])
            next_point = convert_coordinates(path[i+1][0], path[i+1][1])

            while not self.reached_point_building_agent(next_point):
                target_angle = math.atan2(next_point[1] - self.current_position_building_agent[1], 
                                          next_point[0] - self.current_position_building_agent[0])
                angle_diff = self.normalize_angle(target_angle - self.current_orientation_building_agent)

                if abs(angle_diff) > 0.1:
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.3 * angle_diff
                else:
                    vel_msg.angular.z = 0.0
                    vel_msg.linear.x = 0.5

                vel_pub.publish(vel_msg)
                rate.sleep()

            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            vel_pub.publish(vel_msg)
            rate.sleep()

        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        vel_pub.publish(vel_msg)
        
        return "Reached destination"
        
    def move_visitor(self, path: List[Tuple[int, int]]):
        odom_sub = rospy.Subscriber("visitor/odom", Odometry, self.odom_callback_visitor)
        vel_pub = rospy.Publisher('visitor/cmd_vel', Twist, queue_size=10)
        vel_msg = Twist()
        rate = rospy.Rate(10)
        
        while self.current_position_visitor is None:
            rospy.loginfo("Waiting for odometry data...")
            rate.sleep()

        rospy.loginfo("Odometry data received, starting movement...")

        for i in range(len(path) - 1):
            current_point = convert_coordinates(path[i][0], path[i][1])
            next_point = convert_coordinates(path[i+1][0], path[i+1][1])

            while not self.reached_point_visitor(next_point):
                target_angle = math.atan2(next_point[1] - self.current_position_visitor[1], 
                                          next_point[0] - self.current_position_visitor[0])
                angle_diff = self.normalize_angle(target_angle - self.current_orientation_visitor)

                if abs(angle_diff) > 0.1:
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.2 * angle_diff
                else:
                    vel_msg.angular.z = 0.0
                    vel_msg.linear.x = 0.3

                vel_pub.publish(vel_msg)
                rate.sleep()

            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            vel_pub.publish(vel_msg)
            rate.sleep()

        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        vel_pub.publish(vel_msg)
        
        return "Reached destination"

    def reached_point_client(self, point: Tuple[float, float], tolerance: float = 0.1) -> bool:
        if self.current_position_client is None:
            return False
        distance = math.sqrt((point[0] - self.current_position_client[0]) ** 2 + 
                             (point[1] - self.current_position_client[1]) ** 2)
        return distance < tolerance
    
    def reached_point_building_agent(self, point: Tuple[float, float], tolerance: float = 0.1) -> bool:
        if self.current_position_building_agent is None:
            return False
        distance = math.sqrt((point[0] - self.current_position_building_agent[0]) ** 2 + 
                             (point[1] - self.current_position_building_agent[1]) ** 2)
        return distance < tolerance
    
    def reached_point_visitor(self, point: Tuple[float, float], tolerance: float = 0.1) -> bool:
        if self.current_position_visitor is None:
            return False
        distance = math.sqrt((point[0] - self.current_position_visitor[0]) ** 2 + 
                             (point[1] - self.current_position_visitor[1]) ** 2)
        return distance < tolerance

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
        
    def inverse_convert_coordinates(self, x, y):
        return inverse_convert_coordinates(x, y)

