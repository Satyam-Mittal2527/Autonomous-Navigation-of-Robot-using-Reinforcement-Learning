import math
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
import time

from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from ros_gz_interfaces.srv import SetEntityPose

class HardWorldEnvDQN(gym.Env, Node):
    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, "rl_env_node")

        # ROS2 pub/sub
        self.cmd_pub = self.create_publisher(Twist, '/my_robot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/my_robot/scan', self.scan_callback, 10)
        self.path_pub = self.create_publisher(Path, "/robot_path", 10)
        self.marker_pub = self.create_publisher(Marker, "/trajectory_marker", 10)
        self.teleport_client = self.create_client(SetEntityPose, "/world/maze_world/set_pose")

        # INIT STATE SAFETY
        self.x = 0.0
        self.y = 0.0
        self.odom_offset = np.array([0.0, 0.0])

        self.robot_pos = np.array([0.0, 0.0])

        # Marker setup
        self.marker = Marker()
        self.marker.header.frame_id = "odom"
        self.marker.type = Marker.LINE_STRIP
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.05
        self.marker.color.g = 1.0
        self.marker.color.a = 1.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.odom_msg = None
        self.scan_msg = None

        self.step_count = 0
        self.max_steps = 500

        # ✅ DISCRETE ACTION SPACE (DQN)
        self.action_space = gym.spaces.Discrete(5)

        # SAME OBSERVATION
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(26,),
            dtype=np.float32
        )

        self.goal = np.array([7.0, 0.0])
        self.reset_metrics()

    # ======================
    # ACTION MAPPING (IMPORTANT)
    # ======================
    def _map_action(self, action):
        if action == 0:
            return [0.3, 0.0]
        elif action == 1:
            return [0.15, 0.5]
        elif action == 2:
            return [0.15, -0.5]
        elif action == 3:
            return [0.1, 0.0]
        else:
            return [0.0, 0.8]

    # ======================
    # CALLBACKS
    # ======================
    def teleport_robot(self):
        req = SetEntityPose.Request()
        req.entity.name = "robot1"
        req.pose.position.x = -4.3
        req.pose.position.y = 0.0
        req.pose.position.z = 0.2
        req.pose.orientation.w = 1.0

        if self.teleport_client.service_is_ready():
            future = self.teleport_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        self.scan_msg = msg

    # ======================
    # UTILS
    # ======================
    def reset_metrics(self):
        self.collision_count = 0
        self.path_length = 0
        self.prev_pos = None
        self.smoothness = 0
        self.prev_ang_vel = 0
        self.trajectory = []
        self.start_time = time.time()

    def get_robot_position(self):
        return np.array([self.x, self.y])

    def check_collision(self):
        if self.scan_msg is None:
            return False
        return np.min(self.scan_msg.ranges) < 0.2

    def get_lidar(self):
        if self.scan_msg is None:
            return np.ones(24)

        ranges = np.array(self.scan_msg.ranges)
        ranges = np.clip(ranges, 0, 3.5) / 3.5
        return ranges[::15]

    # ======================
    # RESET
    # ======================
    def get_angular_velocity(self):
        #return self.odom_msg.twist.twist.angular.z
        return 0.0
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        print("RESET START")

        self.cmd_pub.publish(Twist())

        self.teleport_robot()
        time.sleep(2.0)

        self.scan_msg = None

        while self.scan_msg is None:
            rclpy.spin_once(self, timeout_sec=0.05)

        self.goal = np.array([7.0, 0.0])

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.reset_metrics()
        self.marker.points = []

        self.step_count = 0
        self.prev_pos = np.array([self.x, self.y])
        self.prev_ang_vel = 0.0

        self.odom_offset = self.get_robot_position().copy()

        obs = self._get_obs()

        print("RESET DONE")
        return obs, {}

    # ======================
    # STEP
    # ======================
    def step(self, action):

        # ✅ FIXED: map discrete → continuous
        linear, angular = self._map_action(action)
        # Smoothness = change in angular velocity
        self.smoothness += abs(angular - self.prev_ang_vel)
        self.prev_ang_vel = angular
        print(f"STEP:{linear}{angular}")
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.01)

        pos = np.array([self.x, self.y]) - self.odom_offset

        if self.prev_pos is None:
            self.prev_pos = pos.copy()

        dist = np.linalg.norm(pos - self.goal)
        print(f"Distance: {dist}")
        obs = self._get_obs()
        lidar = self.get_lidar()

        # VISUALIZATION
        pose = PoseStamped()
        now = self.get_clock().now().to_msg()

        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])

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
        reward += 3.0 * progress

        terminated = False
        truncated = False

        if dist < 1.0:
            print("Goal Reached");
            reward += 100
            terminated = True

        if self.check_collision():
            print("Colllision detectted");
            reward -= 10
            self.collision_count += 1
            truncated = True

        reward -= 0.01

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        # METRICS
        self.path_length += np.linalg.norm(pos - self.prev_pos)

        self.prev_pos = pos.copy()

        info = {}

        if terminated or truncated:
            episode_time = time.time() - self.start_time

            info["is_success"] = 1 if dist < 1.0 else 0

            # collision (binary)
            info["collision_occurred"] = 1 if self.collision_count > 0 else 0

            # path efficiency
            optimal_dist = np.linalg.norm(self.goal)
            info["path_efficiency"] = optimal_dist / (self.path_length + 1e-6)

            # smoothness
            info["smoothness"] = self.smoothness

            # episode duration
            info["episode_time"] = episode_time

        return obs, reward, terminated, truncated, info

    # ======================
    # OBS
    # ======================
    def _get_obs(self):
        lidar = self.get_lidar()
        lidar = lidar * 2.0 - 1.0

        pos = self.get_robot_position()
        goal_vec = self.goal - pos

        norm = np.linalg.norm(goal_vec) + 1e-6
        goal_dir = goal_vec / norm

        return np.concatenate([lidar, goal_dir])
   

