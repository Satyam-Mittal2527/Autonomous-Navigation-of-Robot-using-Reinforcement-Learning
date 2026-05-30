import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, GroupAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():

    pkg_path = get_package_share_directory('my_robot_description')
    urdf_file = os.path.join(pkg_path, 'urdf', 'my_robot.urdf.xacro')
    robot_description = xacro.process_file(urdf_file).toxml()

    # World argument - default is room.sdf
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='room',
        description='World to load: room or hard_world'
    )

    world_name = LaunchConfiguration('world')

    import subprocess
    import sys

    def get_world_file(context, *args, **kwargs):
        name = context.launch_configurations['world']
        return os.path.join(pkg_path, 'worlds', f'{name}.sdf')

    from launch.actions import OpaqueFunction

    def launch_setup(context, *args, **kwargs):
        world = context.launch_configurations.get('world', 'room')
        world_file = os.path.join(pkg_path, 'worlds', f'{world}.sdf')

        return [
            ExecuteProcess(
                cmd=['gz', 'sim', '-r', world_file],
                output='screen'
            ),

            # ROBOT 1
            GroupAction(actions=[
                PushRosNamespace('my_robot'),
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    output='screen',
                    parameters=[{
                        'robot_description': robot_description,
                        'use_sim_time': True,
                        'frame_prefix': 'my_robot/'
                    }]
                ),
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    arguments=[
                        '-name', 'my_robot',
                        '-topic', '/my_robot/robot_description',
                        '-x', '0.0', '-y', '0.5', '-z', '0.1'
                    ],
                    output='screen'
                ),
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    arguments=[
                        '/my_robot/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                        '/my_robot/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                        '/my_robot/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                        '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
                    ],
                    output='screen'
                ),
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name='static_tf_pub_robot1',
                    arguments=['0', '0', '0', '0', '0', '0',
                               'my_robot/odom', 'my_robot/base_footprint']
                ),
            ]),

            # ROBOT 2
            GroupAction(actions=[
                PushRosNamespace('my_robot2'),
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    output='screen',
                    parameters=[{
                        'robot_description': robot_description,
                        'use_sim_time': True,
                        'frame_prefix': 'my_robot2/'
                    }]
                ),
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    arguments=[
                        '-name', 'my_robot2',
                        '-topic', '/my_robot2/robot_description',
                        '-x', '0.0', '-y', '-0.5', '-z', '0.1'
                    ],
                    output='screen'
                ),
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    arguments=[
                        '/my_robot2/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                        '/my_robot2/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                        '/my_robot2/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                    ],
                    output='screen'
                ),
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name='static_tf_pub_robot2',
                    arguments=['0', '0', '0', '0', '0', '0',
                               'my_robot2/odom', 'my_robot2/base_footprint']
                ),
            ]),
        ]

    return LaunchDescription([
        world_arg,
        OpaqueFunction(function=launch_setup),
    ])
