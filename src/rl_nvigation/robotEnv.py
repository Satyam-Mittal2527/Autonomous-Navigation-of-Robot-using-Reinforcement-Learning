import math
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf2_msgs.msg import TFMessage
from ros_gz_interfaces.srv import SetEntityPose
from nav_msgs.msg import Odometry

class HardWorldEnv(gym.Env, Node):

    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, "rl_env_node")

        # ROS2 pub/sub
        self.cmd_pub = self.create_publisher(Twist, '/my_robot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        self.scan_sub = self.create_subscription(LaserScan,'/my_robot/scan',self.scan_callback,10)
        self.path_pub = self.create_publisher(Path, "/robot_path", 10)
        self.marker_pub = self.create_publisher(Marker, "/trajectory_marker", 10)
        self.teleport_client = self.create_client(SetEntityPose,"/world/maze_world/set_pose")
        
        self.robot_pos = np.array([0.0, 0.0])

        self.marker = Marker()
        self.marker.header.frame_id = "odom"
        self.marker.type = Marker.LINE_STRIP   # THIS DRAWS PATH
        self.marker.action = Marker.ADD

        self.marker.scale.x = 0.05  # line thickness

        self.marker.color.r = 0.0
        self.x = 0.0
        self.y = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0
        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"
        self.odom_msg = None
        self.scan_msg = None
        self.step_count = 0
        self.max_steps = 500
        # Action space
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([0.3, 1.0]),
            dtype=np.float32
        )

        # Observation space (LiDAR downsample + dist + angle)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(26,),
            dtype=np.float32
        )

        self.goal = np.array([5.0,0.0])   # your red stand
        self.yaw=0.0
        self.reset_metrics()

    # ======================
    # ROS CALLBACKS
    # ======================
    def teleport_robot(self, x=-0.0, y=0.0, yaw=0.0):

        req = SetEntityPose.Request()

        req.entity.name = "robot1"   # ⚠️ confirm name
        req.pose.position.x = -4.3
        req.pose.position.y = 0.0
        req.pose.position.z = 0.2   # slight lift (important)

        req.pose.orientation.w = 1.0  # no rotation

        if self.teleport_client.service_is_ready():
            future = self.teleport_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            print("🚀 Teleported robot")
        else:
            print("⚠️ Teleport service not ready")
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
    def scan_callback(self, msg):
        self.scan_msg = msg

    # ======================
    # METRICS
    # ======================
    def reset_metrics(self):
        self.collision_count = 0
        self.path_length = 0
        self.prev_pos = None
        self.smoothness = 0
        self.prev_ang_vel = 0
        self.trajectory = []

    # ======================
    # UTILS
    # ======================
    def get_robot_position(self):
        return np.array([self.x, self.y])
    def get_yaw(self):
        q = self.odom_msg.pose.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny, cosy)

    def get_angular_velocity(self):
        #return self.odom_msg.twist.twist.angular.z
        return 0.0
    def check_collision(self):
        return np.min(self.scan_msg.ranges) < 0.2

    def get_lidar(self):
        ranges = np.array(self.scan_msg.ranges)
        ranges = np.clip(ranges, 0, 3.5)/3.5
        return ranges[::15]  # ~24 values

    # ======================
    # RESET
    # ======================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        print("RESET START")
        # STOP ROBOT
        self.cmd_pub.publish(Twist())

        # TELEPORT
        self.teleport_robot(-4.5, 0.0, 0.0)
        pos = np.array([self.x, self.y])
        self.odom_offset = pos.copy()

        
        # IMPORTANT: wait for physics + sensors
        time.sleep(2.0)

        # clear old data
        self.scan_msg = None

        # wait for fresh sensor data
        while self.scan_msg is None:
            rclpy.spin_once(self, timeout_sec=0.05)

        # ✅ Step 3: wait for stable pose (FIXED)
        prev = np.array([self.x, self.y])
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.01)

            curr = np.array([self.x, self.y])
            if np.linalg.norm(curr - prev) < 0.01:
                break

            prev = curr

        # 🔥 extra TF settle (prevents rare noise)
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.01)
        positions = []
        for _ in range(5):
            pos = self.get_robot_position()
            positions.append(pos)
            rclpy.spin_once(self, timeout_sec=0.01)
        

        # 🔥 DEFINE RELATIVE GOAL (FORWARD 4.5m)
        
        self.goal = np.array([5.0, 0.0])
      
        print("REL GOAL:", self.goal)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"
        self.reset_metrics()
        start = time.time()
        self.marker.points = []

        # wait for lidar
        while (self.scan_msg is None):
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.01)

            if time.time() - start > 2.0:
                print("⚠️ Sensor timeout")
                break

        lidar = self.get_lidar()

        if lidar is None:
            print("⚠️ No lidar, using default")
            lidar = np.ones(360)

        if np.min(lidar) < 0.3:
            print("⚠️ Too close to wall, adjusting")

            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.5

            for _ in range(10):
                self.cmd_pub.publish(twist)
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.01)

        self.step_count = 0
        self.prev_pos = None
        self.prev_ang_vel = 0.0

        obs = self._get_obs()

        print("RESET DONE")
        pos = np.array([self.x, self.y])
        self.prev_pos = pos.copy()
        print("NEW RESET CALLED")
        return obs, {}
    # ======================
    # STEP
    # ======================
    def step(self, action):
        print("STEP CALLED",action)
        if self.step_count == 0:
            self.prev_pos = None
            # ===== TERMINATION FLAGS =====
        terminated = False
        truncated = False
    # ===== APPLY ACTION =====
        twist = Twist()
        twist.linear.x = float(np.clip(action[0], 0.0, 0.3))
        twist.angular.z = float(np.clip(action[1], -1.0, 1.0))
        self.cmd_pub.publish(twist)
        #twist.linear.x = 0.0
        #twist.linear.z = 0.0
        #self.cmd_pub.publish(twist)
        for _ in range(10):
            rclpy.spin_once(self,timeout_sec=0.01)
        for _ in range(10):
            rclpy.spin_once(self,timeout_sec=0.01)

        pos = np.array([self.x, self.y]) - self.odom_offset
        if self.prev_pos is None:
            self.prev_pos = pos.copy()
        dist = np.linalg.norm(pos - self.goal)
    # ===== OBSERVATION =====
        obs = self._get_obs()
        lidar = self.get_lidar()
   
        n = len(lidar)
        pose = PoseStamped()
        now = self.get_clock().now().to_msg()
       

        # 🔥 DISTANCE TO GOAL
        print("GOAL:", self.goal)
        print("DIST:", dist)
        print("POS:", pos)
        print(f"[STEP {self.step_count}] x: {pos[0]:.2f}, y: {pos[1]:.2f}")
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = 0.0
        p = Point()
        p.x = float(pos[0])
        p.y = float(pos[1])
        p.z = 0.05

        self.marker.header.stamp = now
        self.marker.points.append(p)

        self.marker_pub.publish(self.marker)
        self.path_msg.poses.append(pose)

        self.path_pub.publish(self.path_msg)
        self.trajectory.append(pos)

    # ===== REWARD =====
        reward = 0.0
        

        prev_dist = np.linalg.norm(self.prev_pos - self.goal)
        progress = prev_dist - dist

        # progress reward
        reward += 3.0 * progress

        # goal
        print(f"Distance: {dist}")
        if dist < 1.0:
            print("Goal Reached")
            reward += 100
            terminated = True

        # collision
        if self.check_collision():
            print("Collision Detected")
            reward -= 10
            self.collision_count += 1
            truncated = True

        # time penalty
        reward -= 0.01

    # ⏱Timeout (VERY IMPORTANT)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

    # ===== METRICS =====
        if self.prev_pos is not None:
            self.path_length += np.linalg.norm(pos - self.prev_pos)

        ang_vel = self.get_angular_velocity()
        self.smoothness += abs(ang_vel - self.prev_ang_vel)
        self.prev_ang_vel = ang_vel

    # ===== RETURN =====
        info = {}
        if terminated or truncated:
            # 1. Trajectory Smoothness (Average change in angular velocity)
            # Lower is better. 0 = perfectly straight or constant turn.
            info["smoothness"] = self.smoothness / max(1, self.step_count)
            start_to_goal = 1.0
    
            # 3. Calculate how far the robot is from the goal RIGHT NOW
            current_to_goal = np.linalg.norm(pos - self.goal)
            # 2. Path Efficiency (Shortest / Actual)
            distance_covered = self.path_length
            actual_path = self.path_length
            direct_dist = np.linalg.norm(self.goal)
            info["path_efficiency"] = max(0.0, distance_covered) / max(0.01, actual_path)
    
            # 3. Collision Rate (Binary for the episode)
            info["collision_occurred"] = 1 if self.collision_count > 0 else 0
    
            # Success Flag
            #info["is_success"] = 1 if terminated and dist < 0.5 else 0
            info["is_success"] = 1 if dist < 1.0 else 0
        self.prev_pos = pos.copy()
        return obs, reward, terminated, truncated, info
    # ======================
    # OBSERVATION
    # ======================
    def state_callback(self, msg):
        #print("CALLBACK TRIGGERED") 
        for model in msg.model:
            print("Model:", model.name)
            # your robot name
            self.x = model.pose.position.x
            self.y = model.pose.position.y

            print(f"x: {self.x:.2f}, y: {self.y:.2f}")
    def _get_obs(self):
        lidar = self.get_lidar()
        lidar = lidar * 2.0 - 1.0   # convert [0,1] → [-1,1]
        pos = self.get_robot_position()
        goal_vec = self.goal - pos

        # normalize goal direction
        norm = np.linalg.norm(goal_vec) + 1e-6
        goal_dir = goal_vec / norm   # [x, y] direction

        return np.concatenate([lidar, goal_dir])
