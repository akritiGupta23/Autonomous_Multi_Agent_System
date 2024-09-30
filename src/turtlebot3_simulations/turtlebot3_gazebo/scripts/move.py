#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import Optional, Tuple, List
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
# from crewai_tools import Tool, BaseTool
import yaml
import cv2
import numpy as np
import heapq
import random
from collections import deque
import math
from tf.transformations import euler_from_quaternion
import threading

class PathPlanningTool(BaseModel):
    start_pose: Optional[Tuple[float, float]] = None
    goal_pose: Optional[Tuple[float, float]] = None
    map_yaml_path: Optional[str] = None
    
    occupancy_grid: Optional[np.ndarray] = None
    resolution: Optional[float] = None
    origin: Optional[Tuple[float, float, float]] = None
    current_pose: Optional[Tuple[float, float]] = None
    current_position_client : Optional[Tuple[float, float]] = None
    current_orientation_client : Optional[float] = None
    current_position_building_agent : Optional[Tuple[float, float]] = None
    current_orientation_building_agent : Optional[float] = None
    current_position_visitor : Optional[Tuple[float, float]] = None
    current_orientation_visitor : Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # if self.map_yaml_path:
        #     self.load_map()
        # self.initialize_ros()

    def load_map(self):
        map_data = self.load_map_yaml(self.map_yaml_path)
        self.occupancy_grid, self.resolution, self.origin = self.map_image_to_grid(map_data)
        return self.occupancy_grid, self.resolution, self.origin

    @staticmethod
    def load_map_yaml(file_path: str) -> dict:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def map_image_to_grid(map_data: dict) -> Tuple[np.ndarray, float, Tuple[float, float, float]]:
        import matplotlib.pyplot as plt

        f = 'map.pgm'

        with open(f, 'rb') as pgmf:
            image = plt.imread(pgmf)
            
        # image = cv2.imread(map_data['image'], cv2.IMREAD_GRAYSCALE)
        resolution = map_data['resolution']
        origin = map_data['origin']
        print(image)
        print(np.unique(image))
        occupancy_grid = np.zeros_like(image)
        # occupancy_grid[image > map_data['occupied_thresh'] * 255] = 100
        # occupancy_grid[image < map_data['free_thresh'] * 255] = 0
        # occupancy_grid[(image >= map_data['free_thresh'] * 255) & (image <= map_data['occupied_thresh'] * 255)] = -1
        occupancy_grid[image == 205] = 100
        occupancy_grid[image == 254] = 100
        occupancy_grid[image == 0] = 0
        # can move only on 100 ...
        # 0 means obstacles
        # 255 means outside the campus
        
        print("occupancy_grid:", occupancy_grid.shape)
        return occupancy_grid, resolution, origin

    # @staticmethod
    # def convert_coordinates(x, y):
    #     # Original range: (0, 0) to (384, 384)
    #     # Target range: (-10, 10) to (10, -10)

    #     # Define the original and target ranges
    #     orig_min, orig_max = 0, 384
    #     target_min, target_max = -10, 10

    #     # Convert coordinates
    #     scaled_x = target_min + (x - orig_min) * (target_max - target_min) / (orig_max - orig_min)
    #     scaled_y = target_max - (y - orig_min) * (target_max - target_min) / (orig_max - orig_min)

    #     return scaled_x, scaled_y

    # @staticmethod
    # def inverse_convert_coordinates(x, y):# -> tuple:
    #     # Target range: (-10, 10) to (10, -10)
    #     # Original range: (0, 0) to (384, 384)

    #     # Define the original and target ranges
    #     orig_min, orig_max = 0, 384
    #     target_min, target_max = -10, 10

    #     # Convert coordinates
    #     orig_x = (x - target_min) * (orig_max - orig_min) / (target_max - target_min) + orig_min
    #     orig_y = (target_max - y) * (orig_max - orig_min) / (target_max - target_min) + orig_min

    #     return orig_x, orig_y
    @staticmethod
    def inverse_convert_coordinates( x , y ) : # x ,  y 10 vale coordinates 
        newx = 19.15 * (10 - y ) 
        newy = 19.15 * (10 + x ) 
        return (newx  , newy ) 

    @staticmethod
    def convert_coordinates( x , y ) : # x , y 384 vale points 
        newx = (y/19.15) - 10
        newy = (10-x/19.15) 
        return (newx , newy)

    @staticmethod
    def bfs_traversal(occupancy_grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        q = deque()
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        q.append([start , [start]])

        move_directions: List[Tuple[float, float]] = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        if(occupancy_grid[start[0]][start[1]] != 100 ) : 
            print("Start point invalid")
            return None
        visited = set()
        visited.add(start)
        while q:
            node , path = q.popleft()

            if(node == goal ) : 
                print("path found")
                return path
            
            for dir in move_directions:
                x = dir[0]
                y = dir[1] 
                newNode = (node[0] + x , node[1] + y )
                if(newNode not in visited and occupancy_grid[node[0] + x][node[1] + y] == 100 ) : 
                    q.append([newNode , path + [newNode]])
                    visited.add(newNode)
        print("path not found")
        return None

    @staticmethod
    def a_star_algorithm(occupancy_grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start = (round(start[0]), round(start[1]))
        goal = (round(goal[0]), round(goal[1]))
        
        print(start)
        print(goal)
        
        steps = 0 
        def euclideanDist(start , end ) : 
            return ((start[0] - end[0])**2 + (start[1] - end[1])**2) ** 0.5
        def heuristic_function( start , end ) : 
            return abs(start[0] - end[0]) + abs(start[1] - end[1])

        pq = []
        heapq.heappush( pq , ( 0 , start , [start] ))
        move_directions: List[Tuple[float, float]] = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        if(occupancy_grid[start[0]][start[1]] != 100 ) : 
            print("Start point invalid")
            return None
        visited = set()
        visited.add(start)
        while pq:
            steps += 1
            cost , node , path = heapq.heappop(pq)

            if(node == goal ) : 
                print(f"path found in {steps} steps")
                return path
            
            for dir in move_directions:
                x = dir[0]
                y = dir[1] 
                newNode = (node[0] + x , node[1] + y )
                # print(newNode)
                # print(occupancy_grid[node[0] + x][node[1] + y])
                if(newNode not in visited and occupancy_grid[node[0] + x][node[1] + y] == 100 ) : 
                    # q.append([newNode , path + [newNode]])
                    h = heuristic_function(newNode , goal)
            
                    heapq.heappush(pq , (cost + euclideanDist(node , newNode ) + h , newNode , path + [newNode]))
                    visited.add(newNode)
        print("path not found")
        return None
    
    def odom_callback_client_agent(self, msg):
        # Extract position
        self.current_position_client = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # Extract orientation in quaternion and convert to Euler angles (roll, pitch, yaw)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_orientation_client = euler_from_quaternion(orientation_list)
        
    def odom_callback_building_agent(self, msg):
        # Extract position
        self.current_position_building_agent = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # Extract orientation in quaternion and convert to Euler angles (roll, pitch, yaw)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_orientation_building_agent = euler_from_quaternion(orientation_list)
        
    def odom_callback_visitor(self, msg):
        # Extract position
        self.current_position_visitor = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # Extract orientation in quaternion and convert to Euler angles (roll, pitch, yaw)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_orientation_visitor = euler_from_quaternion(orientation_list)
    
    @staticmethod
    def callback(msg):
        global follower_pub
        twist = Twist()
        # Copy the linear and angular velocities from the leader
        twist.linear.x = msg.linear.x
        twist.angular.z = msg.angular.z
        follower_pub.publish(twist)

    def move_client_agent(self, path: List[Tuple[int, int]]):
        odom_sub = rospy.Subscriber("client_agent/odom", Odometry, self.odom_callback_client_agent)
        print(odom_sub)
        
        vel_pub = rospy.Publisher('client_agent/cmd_vel', Twist, queue_size=10)  # Publisher for robot velocity commands
        vel_msg = Twist()

        rate = rospy.Rate(10)  # 10 Hz loop rate
        
        while self.current_position_client is None:
            rospy.loginfo("Waiting for odometry data...")
            rate.sleep()

        rospy.loginfo("Odometry data received, starting movement...")

        for i in range(len(path) - 1):
        # for i in range(1):
            # print(f"Current Postion : {path[i][0] , path[i][1]}")
            # print(f"Next Postioon : {path[i+1][0] , path[i+1][1]}")
            current_point = self.convert_coordinates(path[i][0], path[i][1])
            next_point = self.convert_coordinates(path[i+1][0], path[i+1][1])
            
            print(current_point)
            print(self.current_position_client)

                        # Keep moving towards the next point until it's reached
            while not self.reached_point_client(next_point):
                # Calculate the distance and angle to the next point
                target_distance = math.sqrt((next_point[0] - self.current_position_client[0]) ** 2 + 
                                            (next_point[1] - self.current_position_client[1]) ** 2)
                target_angle = math.atan2(next_point[1] - self.current_position_client[1], 
                                          next_point[0] - self.current_position_client[0])

                # Calculate the difference in orientation
                angle_diff = self.normalize_angle(target_angle - self.current_orientation_client)

                # Rotate the robot towards the target
                if abs(angle_diff) > 0.1:  # Threshold for minimal angle difference
                    vel_msg.linear.x = 0.0  # Stop linear movement during rotation
                    vel_msg.angular.z = 0.2 * angle_diff  # Proportional rotation adjustment
                else:
                    vel_msg.angular.z = 0.0  # Stop rotation once aligned
                    vel_msg.linear.x = 0.3  # Move forward

                vel_pub.publish(vel_msg)
                rate.sleep()

            # Stop the robot after reaching the next point
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            vel_pub.publish(vel_msg)
            rate.sleep()

        # Stop the robot at the goal
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        vel_pub.publish(vel_msg)
        
        return "Reached destination"
        
    def move_building_agent(self, path: List[Tuple[int, int]]):
        odom_sub = rospy.Subscriber("building_agent/odom", Odometry, self.odom_callback_building_agent)
        print(odom_sub)
        
        vel_pub = rospy.Publisher('building_agent/cmd_vel', Twist, queue_size=10)  # Publisher for robot velocity commands
        vel_msg = Twist()

        rate = rospy.Rate(10)  # 10 Hz loop rate
        
        while self.current_position_building_agent is None:
            rospy.loginfo("Waiting for odometry data...")
            rate.sleep()

        rospy.loginfo("Odometry data received, starting movement...")

        for i in range(len(path) - 1):
            current_point = self.convert_coordinates(path[i][0], path[i][1])
            next_point = self.convert_coordinates(path[i+1][0], path[i+1][1])
            print("Bulding Agent")
            print(current_point , (path[i][0] , path[i][1]))
            print(self.current_position_client)

                        # Keep moving towards the next point until it's reached
            while not self.reached_point_building_agent(next_point):
                # Calculate the distance and angle to the next point
                target_distance = math.sqrt((next_point[0] - self.current_position_building_agent[0]) ** 2 + 
                                            (next_point[1] - self.current_position_building_agent[1]) ** 2)
                target_angle = math.atan2(next_point[1] - self.current_position_building_agent[1], 
                                          next_point[0] - self.current_position_building_agent[0])

                # Calculate the difference in orientation
                angle_diff = self.normalize_angle(target_angle - self.current_orientation_building_agent)

                # Rotate the robot towards the target
                if abs(angle_diff) > 0.1:  # Threshold for minimal angle difference
                    vel_msg.linear.x = 0.0  # Stop linear movement during rotation
                    vel_msg.angular.z = 0.3 * angle_diff  # Proportional rotation adjustment
                else:
                    vel_msg.angular.z = 0.0  # Stop rotation once aligned
                    vel_msg.linear.x = 0.5  # Move forward

                vel_pub.publish(vel_msg)
                rate.sleep()

            # Stop the robot after reaching the next point
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            vel_pub.publish(vel_msg)
            rate.sleep()

        # Stop the robot at the goal
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        vel_pub.publish(vel_msg)
        
        return "Reached destination"
        
    def move_visitor(self, path: List[Tuple[int, int]]):
        odom_sub = rospy.Subscriber("visitor/odom", Odometry, self.odom_callback_visitor)
        print(odom_sub)
        
        vel_pub = rospy.Publisher('visitor/cmd_vel', Twist, queue_size=10)  # Publisher for robot velocity commands
        vel_msg = Twist()

        rate = rospy.Rate(10)  # 10 Hz loop rate
        
        while self.current_position_visitor is None:
            rospy.loginfo("Waiting for odometry data...")
            rate.sleep()

        rospy.loginfo("Odometry data received, starting movement...")

        for i in range(len(path) - 1):
            current_point = self.convert_coordinates(path[i][0], path[i][1])
            next_point = self.convert_coordinates(path[i+1][0], path[i+1][1])
            print("Visitor")
            print(current_point)
            print(self.current_position_client)

                        # Keep moving towards the next point until it's reached
            while not self.reached_point_visitor(next_point):
                # Calculate the distance and angle to the next point
                target_distance = math.sqrt((next_point[0] - self.current_position_visitor[0]) ** 2 + 
                                            (next_point[1] - self.current_position_visitor[1]) ** 2)
                target_angle = math.atan2(next_point[1] - self.current_position_visitor[1], 
                                          next_point[0] - self.current_position_visitor[0])

                # Calculate the difference in orientation
                angle_diff = self.normalize_angle(target_angle - self.current_orientation_visitor)

                # Rotate the robot towards the target
                if abs(angle_diff) > 0.1:  # Threshold for minimal angle difference
                    vel_msg.linear.x = 0.0  # Stop linear movement during rotation
                    vel_msg.angular.z = 0.2 * angle_diff  # Proportional rotation adjustment
                else:
                    vel_msg.angular.z = 0.0  # Stop rotation once aligned
                    vel_msg.linear.x = 0.3  # Move forward

                vel_pub.publish(vel_msg)
                rate.sleep()

            # Stop the robot after reaching the next point
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            vel_pub.publish(vel_msg)
            rate.sleep()

        # Stop the robot at the goal
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        vel_pub.publish(vel_msg)
        
        return "Reached destination"

    def reached_point_client(self, point: Tuple[float, float], tolerance: float = 0.1) -> bool:
        """Check if the robot has reached the target point within a certain tolerance."""
        if self.current_position_client is None:
            return False  # If odometry is not available yet
        distance = math.sqrt((point[0] - self.current_position_client[0]) ** 2 + 
                             (point[1] - self.current_position_client[1]) ** 2)
        return distance < tolerance
    
    def reached_point_building_agent(self, point: Tuple[float, float], tolerance: float = 0.1) -> bool:
        """Check if the robot has reached the target point within a certain tolerance."""
        if self.current_position_building_agent is None:
            return False  # If odometry is not available yet
        distance = math.sqrt((point[0] - self.current_position_building_agent[0]) ** 2 + 
                             (point[1] - self.current_position_building_agent[1]) ** 2)
        return distance < tolerance
    
    def reached_point_visitor(self, point: Tuple[float, float], tolerance: float = 0.1) -> bool:
        """Check if the robot has reached the target point within a certain tolerance."""
        if self.current_position_visitor is None:
            return False  # If odometry is not available yet
        distance = math.sqrt((point[0] - self.current_position_visitor[0]) ** 2 + 
                             (point[1] - self.current_position_visitor[1]) ** 2)
        return distance < tolerance

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize an angle to the range [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class Client_Follower(BaseModel):
    visitor_vel_pub: Optional[rospy.Publisher] = None
    cmd_vel_sub: Optional[rospy.Subscriber] = None
    stop_follower: bool = False  # Shared flag to stop follower thread

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize ROS node
        self.cmd_vel_sub = rospy.Subscriber('client_agent/cmd_vel', Twist, self.callback)
        self.visitor_vel_pub = rospy.Publisher('visitor/cmd_vel', Twist, queue_size=10)
        rospy.loginfo("Visitor node initialized and listening to client_agent/cmd_vel.")

    def callback(self, msg: Twist):
        # Relay the velocity commands from client_agent to visitor
        if not self.stop_follower:  # Only relay if not stopped
            rospy.loginfo("Relaying velocity command from client_agent to visitor.")
            self.visitor_vel_pub.publish(msg)
        
    def run(self):
        # Use a loop with rate.sleep() instead of rospy.spin() to allow stopping
        rate = rospy.Rate(20)  # 10 Hz
        while not self.stop_follower:
            rate.sleep()  # Check every 0.1 seconds for stop_follower flag
        rospy.loginfo("Client follower thread stopped.")
            
class Client_building_Follower(BaseModel):
    visitor_vel_pub: Optional[rospy.Publisher] = None
    cmd_vel_sub: Optional[rospy.Subscriber] = None
    stop_follower: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize ROS node
        self.cmd_vel_sub = rospy.Subscriber('building_agent/cmd_vel', Twist, self.callback)
        self.visitor_vel_pub = rospy.Publisher('client_agent/cmd_vel', Twist, queue_size=10)
        rospy.loginfo("Visitor node initialized and listening to building_agent/cmd_vel.")

    def callback(self, msg: Twist):
        # Relay the velocity commands from building_agent to visitor
        if not self.stop_follower:  # Only relay if not stopped
            # rospy.loginfo("Relaying velocity command from building_agent to visitor.")
            self.visitor_vel_pub.publish(msg)
        
    def run(self):
        # Use a loop with rate.sleep() instead of rospy.spin() to allow stopping
        rate = rospy.Rate(20)  # 10 Hz
        while not self.stop_follower:
            rate.sleep()  # Check every 0.1 seconds for stop_follower flag
        rospy.loginfo("Building follower thread stopped.")


def client_run_follower(follower):
    follower.run()

def building_agent_run_follower(follower):
    follower.run()

def run_client_agent(obj, path, follower):
    result = obj.move_client_agent(path)
    print(result)
    # Signal the follower to stop when the client reaches its goal
    if result == "Reached destination":
        follower.stop_follower = True

def run_building_agent(obj, path, follower, follower2):
    result = obj.move_building_agent(path)
    print(result)
    # Signal the follower to stop when the building agent reaches its goal
    if result == "Reached destination":
        follower.stop_follower = True
        follower2.stop_follower = True


rospy.init_node('path_planning_node', anonymous=True)

# Initialize your path planning object and run your code as before
obj = PathPlanningTool(start_pose=(0.0, 0.0), goal_pose=(10.0, 10.0), map_yaml_path='/home/akriti/catkin_ws/map.yaml')
occupancy_grid, _, _ = obj.load_map()
print(1)
print(np.count_nonzero(occupancy_grid == 100))

### Step 1: Client + Visitor Follower Segment ###
#print("Client + Visitor system operating")
#start_pose = obj.inverse_convert_coordinates(6, 1)
#goal_pose = obj.inverse_convert_coordinates(0, 0)
#path = obj.a_star_algorithm(occupancy_grid, start_pose, goal_pose)
#print(path)

#follower = Client_Follower()

# Run follower and client_agent in parallel using threads
#client_thread = threading.Thread(target=run_client_agent, args=(obj, path, follower))
#client_follower_thread = threading.Thread(target=client_run_follower, args=(follower,))

# Start both threads
#client_thread.start()
#client_follower_thread.start()

# Wait for both threads to finish
#client_thread.join()
#client_follower_thread.join()

#print("Hey Visitor!! You have reached the building")

### Step 3: Building Agent + Visitor Segment ###
print("Building agent + visitor system operating")
start_pose = obj.inverse_convert_coordinates(0, 0.5)

room = {'LAB1': (-4,-5)}
visitor_goal = input("Hey Visitor!! I am Building agent. I will escort you to the desired room. You want to go to LAB1 or LAB2 or Office or Discussion Room")

goal_pose = obj.inverse_convert_coordinates(room[visitor_goal][0], room[visitor_goal][1])


print("Start Postiion : " , start_pose )
print( "Goal Position : " , goal_pose )


path = obj.a_star_algorithm(occupancy_grid, start_pose, goal_pose)
# path = obj.bfs_traversal(occupancy_grid, start_pose, goal_pose)
print(path)

follower = Client_building_Follower()
follower2 = Client_Follower()

# Run follower and building_agent in parallel using threads
building_agent_thread = threading.Thread(target=run_building_agent, args=(obj, path, follower,follower2,))
building_agent_follower_thread = threading.Thread(target=building_agent_run_follower, args=(follower,))
client_follow_builder = threading.Thread(target=client_run_follower, args=(follower2,))

# Start both threads
building_agent_thread.start()
building_agent_follower_thread.start()
client_follow_builder.start()

# Wait for both threads to finish
building_agent_thread.join()
building_agent_follower_thread.join()
client_follow_builder.join()

print("Kudos Visitor!! You have reached your final location")
